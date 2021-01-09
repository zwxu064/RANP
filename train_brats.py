import os, torch, time, math, random, copy
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from third_party.unet.model import UNet3D, ResidualUNet3D
from third_party.miccai.Dataloader.dataloader import BraTSDataset, multi_slice_viewer
from pruning.pytorch_snip.prune import pruning, apply_prune_mask_3dunet, apply_hidden_mask
from pruning.pytorch_snip.prune import do_statistics_model, dump_neuron_per_layer
from pruning_related import refine_model
from third_party.miccai.metrics import dice
from third_party.BraTS2018_tumor.data.singlepath import SingleData
from third_party.BraTS2018_tumor.data.data_utils import init_fn
from third_party.BraTS2018_tumor.models.criterions import cross_entropy_dice
from third_party.thop.thop.profile import profile
from aux.utils import model_transfer_last_layer, get_model_state_dict, load_optimizer_state_dict
from aux.utils import check_dir, weight_init, AverageMeter, write_to_tensorboard
from configs import set_config
import matplotlib.pyplot as plt


def calculate_accuracy(estimation, target, eps=1e-8):
  valid_area_est = (estimation > 0).float()
  valid_area_tar = (target > 0).float()

  common_part = valid_area_est * valid_area_tar
  intersection = (estimation[common_part.byte()] == target[common_part.byte()]).double().sum()
  union = valid_area_est.double().sum() + valid_area_tar.double().sum() - common_part.double().sum()
  iou = intersection / (union + eps)

  return iou


def adjust_learning_rate(optimizer, epoch):
  if epoch + 1 in {150, 250}:
    for param_group in optimizer.param_groups:
      param_group['lr'] *= 0.1


def train(model, dataset, optimizer, writer, args):
  model.train()
  train_loss = AverageMeter()
  train_acc = AverageMeter()
  scores = AverageMeter()
  adjust_learning_rate(optimizer, args.current_epoch)
  time_start = time.time()
  current_iter = args.current_epoch * len(dataset)

  for idx, data in enumerate(dataset):
    if isinstance(data, dict):
      input, gt = data['x'], data['y']
    else:
      input, gt = data[0], data[1]

    actual_batch = input.size(0)

    if args.enable_cuda:
      input, gt = input.cuda(), gt.cuda()

    # For padding area, gt will be fed by 0 but not ignored in criterion
    valid_area = (gt != args.ignore_index)
    valid_area_flatten = valid_area.reshape(actual_batch, -1)

    optimizer.zero_grad()
    prediction = model(input)  # prediction-score:(batch,5,128,128,128)
    prediction = prediction.reshape(actual_batch, args.n_class, -1)
    gt = gt.reshape(actual_batch, -1)

    if args.alpha is None:
      loss = criterion(prediction, gt, ignore_index=args.ignore_index)
    else:
      loss = criterion(prediction, gt, args.alpha, ignore_index=args.ignore_index)

    loss.backward()
    optimizer.step()

    seg_estimated = prediction.argmax(1)
    accuracies = calculate_accuracy(seg_estimated[valid_area_flatten], gt[valid_area_flatten])

    # New metrics
    for b_idx in range(actual_batch):
      pred_one = prediction[b_idx]
      gt_one = gt[b_idx]
      pred_one = pred_one.argmax(0)
      score_one = dice(pred_one[valid_area_flatten[b_idx]].cpu().numpy(), gt_one[valid_area_flatten[b_idx]].cpu().numpy())
      scores.update(np.array(score_one))

    train_loss.update(loss.item(), actual_batch)
    train_acc.update(accuracies, actual_batch)
    print('Epoch: {}, batch: {}, train loss: {:.4f}, train acc: {:.4f}; ' \
          'whole: {:.4f}, core: {:.4f}, enhance: {:.4f}.' \
          .format(args.current_epoch, idx, train_loss.avg, train_acc.avg,
                  scores.avg[0], scores.avg[1], scores.avg[2]))

    current_iter += 1
    train_iter_result = {'mode': 'train', 'batch_iter': current_iter, 'train_loss_iter': train_loss.avg}
    write_to_tensorboard('scalar', writer, train_iter_result)

  duration = time.time() - time_start
  print('Epoch: {}, train loss: {:.4f}, train acc: {:.4f}; ' \
        'whole: {:.4f}, core: {:.4f}, enhance: {:.4f}; time: {:.4f}s' \
        .format(args.current_epoch, train_loss.avg, train_acc.avg,
                scores.avg[0], scores.avg[1], scores.avg[2], duration))

  train_result = {'mode': 'train', 'epoch': args.current_epoch,
                  'train_loss': train_loss.avg, 'train_acc': train_acc.avg,
                  'train_whole': scores.avg[0], 'train_core': scores.avg[1],
                  'train_enhance': scores.avg[2]}
  write_to_tensorboard('scalar', writer, train_result)


def valid(model, dataset, writer, names, args):
  model.eval()
  valid_loss = AverageMeter()
  valid_acc = AverageMeter()
  scores = AverageMeter()
  time_start = time.time()

  for idx, data in enumerate(dataset):
    if isinstance(data, dict):
      input, gt = data['x'], data['y']
    else:
      input, gt = data[0], data[1]

    actual_batch = input.size(0)

    if args.enable_cuda:
      input, gt = input.cuda(), gt.cuda()

    valid_area = (gt != args.ignore_index)
    valid_area_flatten = valid_area.reshape(actual_batch, -1)

    prediction = model(input)
    prediction = prediction.reshape(actual_batch, args.n_class, -1)
    gt = gt.reshape(actual_batch, -1)

    if args.alpha is None:
      loss = criterion(prediction, gt, ignore_index=args.ignore_index)
    else:
      loss = criterion(prediction, gt, args.alpha, ignore_index=args.ignore_index)

    seg_estimated = nn.Softmax(dim=1)(prediction).argmax(1)
    accuracies = calculate_accuracy(seg_estimated[valid_area_flatten], gt[valid_area_flatten])

    # New metrics
    for b_idx in range(actual_batch):
      pred_one = prediction[b_idx]
      gt_one = gt[b_idx]
      pred_one = pred_one.argmax(0)
      score_one = dice(pred_one[valid_area_flatten[b_idx]].cpu().numpy(), gt_one[valid_area_flatten[b_idx]].cpu().numpy())
      scores.update(np.array(score_one))

    valid_acc.update(accuracies, actual_batch)
    valid_loss.update(loss.item(), actual_batch)
    print('Epoch: {}, batch: {}, valid loss: {:.4f}, valid acc: {:.4f}; ' \
          'whole: {:.4f}, core: {:.4f}, enhance: {:.4f}, name: {}.' \
          .format(args.current_epoch, idx, valid_loss.avg, valid_acc.avg,
                  scores.avg[0], scores.avg[1], scores.avg[2], names[idx]))

  duration = time.time() - time_start
  valid_result = {'mode': 'valid', 'epoch': args.current_epoch,
                  'valid_loss': valid_loss.avg, 'valid_acc': valid_acc.avg,
                  'valid_whole': scores.avg[0], 'valid_core': scores.avg[1],
                  'valid_enhance': scores.avg[2]}
  write_to_tensorboard('scalar', writer, valid_result)

  print('Epoch: {}, valid loss: {:.4f}, valid acc: {:.4f}; '
        'whole: {:.4f}, core: {:.4f}, enhance: {:.4f}; time: {:.4f}s' \
        .format(args.current_epoch, valid_loss.avg, valid_acc.avg,
                scores.avg[0], scores.avg[1], scores.avg[2], duration))

  return valid_acc.avg, valid_loss.avg


def test(model, dataset, args):
  model.eval()

  for idx, data in enumerate(dataset):
    if isinstance(data, dict):
      inputs, gt = data['x'], data['y']
    else:
      inputs, gt = data[0], data[1]

    if args.enable_cuda:
      inputs, gt = inputs.cuda(), gt.cuda()

    prediction = model(inputs)
    name = data[2][0]
    inputs = np.transpose(inputs[0].cpu().numpy(), (0, 3, 2, 1))
    gt = np.transpose(gt[0].cpu().numpy(), (2, 1, 0))
    prediction = np.transpose(np.argmax(prediction[0].cpu().numpy(), 0), (2, 1, 0))
    viz_data = [inputs[0], inputs[1], inputs[2], inputs[3], gt, prediction]
    viz_disp = ['t1', 't1ce', 't2', 'flair', 'gt', 'pred']

    print(name)
    multi_slice_viewer('manual', viz_data, viz_disp)
    plt.show()

    index = int(input('please input a number:\n'))
    if index >= 0:
      for k in range(6):
        plt.figure()
        plt.imshow(viz_data[k][index])
        plt.axis(False)
        plt.savefig('viz_figures/brats/{}_{}_index{}.eps'.format(name, viz_disp[k], index), format='eps')
        plt.close()


if __name__ == '__main__':
  # Set env
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.enabled = True

  # Set seed
  seed = 2001
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  # Config
  args = set_config()
  enable_dump_neuron_per_layer = False

  if not isinstance(args.years, list):
    args.years = [args.years]

  enable_hidden_sum = False
  assert not enable_hidden_sum

  if args.checkpoint_dir is not None:
    if args.enable_train:
      args.checkpoint_dir = os.path.join(args.checkpoint_dir, 'sz{}_d{}_s{}'.format(args.spatial_size, args.dim, args.scale))

      if args.number_of_fmaps != 4:
        args.checkpoint_dir = args.checkpoint_dir + '_depth{}'.format(args.number_of_fmaps)

      if args.enable_neuron_prune:
        if (args.layer_sparsity_list is not None) and (args.layer_sparsity_list > 0):
          if isinstance(args.layer_sparsity_list, list):
            args.checkpoint_dir = args.checkpoint_dir + '_layerpruninglist{}'.format(args.layer_sparsity_list[0])
          else:
            args.checkpoint_dir = args.checkpoint_dir + '_layerpruning{}'.format(args.layer_sparsity_list)
        else:
          args.checkpoint_dir = args.checkpoint_dir + '_pruning{}'.format(args.neuron_sparsity)

      if args.enable_hidden_layer_prune:
        args.checkpoint_dir = args.checkpoint_dir + '_hidden{}'.format(args.hidden_layer_sparsity)

      if args.enable_param_prune:
        args.checkpoint_dir = args.checkpoint_dir + '_param{}'.format(args.param_sparsity)

    args.event_dir = os.path.join(args.checkpoint_dir, 'event')

    if args.enable_train:
      args.event_dir = args.event_dir + '_train'
    elif args.enable_test:
      args.event_dir = args.event_dir + '_test'

    args.model_dir = os.path.join(args.checkpoint_dir, 'model')
    check_dir(args.checkpoint_dir)
    check_dir(args.event_dir)
    check_dir(args.model_dir) if args.enable_train else None

  # Model
  if os.path.exists(args.load_transfer_model_path):
    args.enable_neuron_prune = False
    args.enable_hidden_layer_prune = False
    args.enable_param_prune = False
    print('====> Load transfer model from {}'.format(args.load_transfer_model_path))
    model = torch.load(args.load_transfer_model_path)
    model = model_transfer_last_layer(model, 4, args.n_class, args.weight_init)
  else:
    if True:
      in_channels, out_channels, final_sigmoid, f_maps = 4, args.n_class, False, 32  # f_map=64
      model = UNet3D(in_channels,
                     out_channels,
                     final_sigmoid,
                     f_maps=f_maps,
                     layer_order='cbr',
                     num_groups=4,
                     enable_prob=False,
                     number_of_fmaps=args.number_of_fmaps,
                     enable_deepmodel_pooling=args.enable_deepmodel_pooling,
                     width=args.width,
                     res_type=args.res_type)
    else:
      model = Unet(c=4, n=16, dropout=0.3, norm='gn', num_classes=5)

  weight_init(model, mode=args.weight_init)

  # Optimizer
  if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay, amsgrad=True)
  else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay, nesterov=True)

  # Loss
  criterion = cross_entropy_dice

  if args.enable_cuda:
    model = model.cuda()

  # Tensorboard
  writer = SummaryWriter(args.event_dir)

  # Datasets
  args.data_dir = os.path.join(args.data_dir, '2018/MICCAI_BraTS_2018_Data_Training')
  args.train_list = 'train_0.txt'
  args.valid_list = 'valid_0.txt'

  # original size: 240*240*155
  # (batch,h,w,d,channels) in dataset, (batch,channels,h,w,d) in dataloader
  # Pad:(batch,h,w,d,c)
  train_transforms = 'Compose([Pad((0, {}, {}, {}, 0), fill_v=[0, {}]), RandCrop({}), NumpyType((np.float32, np.int64))])' \
                     .format(max(0, args.spatial_size - 240),
                             max(0, args.spatial_size - 240),
                             max(0, args.spatial_size - 155),
                             args.ignore_index,
                             args.spatial_size)
  valid_transforms = 'Compose([Pad((0, {}, {}, {}, 0), fill_v=[0, {}]), CenterCrop({}), NumpyType((np.float32, np.int64))])' \
                      .format(max(0, args.valid_spatial_size - 240),
                              max(0, args.valid_spatial_size - 240),
                              max(0, args.valid_spatial_size - 155),
                              args.ignore_index,
                              args.valid_spatial_size)

  train_list = os.path.join(args.data_dir, args.train_list)
  valid_list = os.path.join(args.data_dir, args.valid_list)

  train_set = SingleData(train_list, root=args.data_dir, for_train=True, transforms=train_transforms)
  train_dataloader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                                num_workers=8, pin_memory=True, worker_init_fn=init_fn)

  valid_set = SingleData(valid_list, root=args.data_dir, for_train=False, transforms=valid_transforms)
  valid_dataloader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

  #
  model_full = copy.deepcopy(model)
  profile_input = torch.randn(1, 4, args.spatial_size, args.spatial_size, args.spatial_size).cuda()
  flops_full, params_full, memory_full, resource_list = profile(model_full.cuda(), inputs=(profile_input,),
                                                                verbose=False,
                                                                resource_list_type=args.resource_list_type)
  print('Full model, flops: {:.4f}G, params: {:.4f}KB, memory: {:.4f}MB' \
        .format(flops_full / 1e9, params_full * 4 / 1024, memory_full * 4 / (1024 ** 2)))
  del model_full

  # Prune including kernels and hidden layers
  if args.enable_neuron_prune or args.enable_hidden_layer_prune or args.enable_param_prune:
    spatial_size_org = args.spatial_size
    dim_org = args.dim
    args.spatial_size = args.prune_spatial_size
    args.dim = args.prune_spatial_size
    grad_mode = 'raw' if args.enable_raw_grad else 'abs'

    if args.weight_init == 'xn':
      file_path = 'data/brats/brats2018_kernel_hidden_prune_grad_sz{}_dim{}_'\
                  'scale{}_fmap{}_depth{}_width{}_{}.npy'. \
        format(args.prune_spatial_size, args.prune_spatial_size, 1,
               f_maps, args.number_of_fmaps, args.width, grad_mode)
    else:
      file_path = 'data/brats/brats2018_kernel_hidden_prune_grad_sz{}_dim{}_'\
                  'scale{}_fmap{}_depth{}_width{}_init{}_{}.npy'. \
        format(args.prune_spatial_size, args.prune_spatial_size, 1,
               f_maps, args.number_of_fmaps, args.width,
               args.weight_init, grad_mode)

    assert (args.batch == 1 if (not os.path.exists(file_path)) else True)

    outputs = pruning(file_path, model, train_dataloader, criterion, args,
                      enable_3dunet=True, enable_hidden_sum=enable_hidden_sum,
                      width=args.width, resource_list=resource_list)
    assert outputs[0] == 0
    neuron_mask_clean, hidden_mask = outputs[1], outputs[2]

    if args.enable_neuron_prune or args.enable_param_prune:
      n_params_org, n_neurons_org = do_statistics_model(model)
      new_model = refine_model(model, neuron_mask_clean, enable_3dunet=True,
                               width=args.width)

      if enable_dump_neuron_per_layer:
        dump_neuron_per_layer(copy.deepcopy(model), copy.deepcopy(new_model))

      del model
      model = new_model.cpu()
      weight_init(model, mode=args.weight_init)
      model = model.cuda()
      n_params_refined, n_neurons_refined = do_statistics_model(model)
      print('Statistics, org, params: {}, neurons: {}; refined, '\
            'params: {} ({:.4f}%), neurons: {} ({:.4f}%)' \
            .format(n_params_org, n_neurons_org,
                    n_params_refined, n_params_refined * 100 / n_params_org,
                    n_neurons_refined, n_neurons_refined * 100 / n_neurons_org))

      idx = 0
      for key, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv3d,
                              nn.ConvTranspose2d, nn.ConvTranspose3d)):
          # print(idx, layer.weight.data.size())
          idx += 1

      # For new model
      if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.w_decay)
      else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.w_decay, nesterov=True)

    if args.enable_hidden_layer_prune:
      apply_hidden_mask(model, hidden_mask, enable_hidden_sum=enable_hidden_sum)

    args.spatial_size = spatial_size_org
    args.dim = dim_org

  # Resume
  if (args.model_dir is not None) and (args.resume_epoch >= 0):
    resume_path = os.path.join(args.model_dir,
                               'model_epoch{}.pth'.format(args.resume_epoch))

    if os.path.exists(resume_path):
      print('Resume from epoch: {}'.format(args.resume_epoch))
      checkpoint = torch.load(resume_path,
                              map_location=lambda storage, loc: storage)
      model.load_state_dict(checkpoint['model'])
      # optimizer from parallel model
      optimizer = load_optimizer_state_dict(checkpoint['optimizer'],
                                            optimizer,
                                            enable_cuda=args.enable_cuda)
    else:
      print('Resume epoch: {} failed, start from 0'.format(args.resume_epoch))
      args.resume_epoch = -1

  # Calculate Flops, Params, Memory
  model_flops = copy.deepcopy(model)
  flops, params, memory, _ = profile(model_flops.cuda(),
                                     inputs=(profile_input,),
                                     verbose=False,
                                     enable_layer_neuron_display=args.enable_layer_neuron_display)
  print('New model, flops: {:.4f}G, params: {:.4f}MB, memory: {:.4f}MB' \
        .format(flops / 1e9, params * 4 / (1024 ** 2), memory * 4 / (1024 ** 2)))
  del model_flops, profile_input

  print(args)

  if (args.save_transfer_model_path != '') and (not os.path.exists(args.save_transfer_model_path)):
    print('====> Save transfer model to {}'.format(args.save_transfer_model_path))
    torch.save(model.cpu(), args.save_transfer_model_path)
    exit()

  # Train and valid
  if args.enable_train:
    model = nn.DataParallel(model) if ((args.batch > 1) and args.enable_cuda) else model

    best_iou = 0
    train_duration, valid_duration = 0, 0
    print('Start of training ...')

    for epoch in range(args.resume_epoch+1, args.epoch):
      args.current_epoch = epoch
      time_start = time.time()
      train(model, train_dataloader, optimizer, writer, args)
      train_duration += time.time() - time_start
      torch.cuda.empty_cache()

      if epoch >= args.valid_min_epoch:
        with torch.no_grad():
          time_start = time.time()
          mean_iou, valid_loss = valid(model, valid_dataloader, writer, valid_set.names, args)
          valid_duration += time.time() - time_start
          torch.cuda.empty_cache()

          # if (mean_iou > best_iou) and (args.model_dir is not None):
          if (args.model_dir is not None) and ((epoch % 1 == 0) or (epoch == args.epoch - 1)):
            best_iou = mean_iou
            filename = 'model_epoch{}.pth'.format(epoch)
            model_state_dict = get_model_state_dict(model)  # parallel model
            torch.save({'epoch': epoch, 'loss': valid_loss,
                        'model': model_state_dict, 'optimizer': optimizer.state_dict()},
                        os.path.join(args.model_dir, filename))

    print('End of training, train time {:.4f} h, valid time {:.4f} h' \
          .format(train_duration / 3600, valid_duration / 3600))

  # Test
  if args.enable_test:
    if args.resume_path is None:
      if (args.test_epoch is None) or (args.test_epoch < 0):
        resume_paths = [os.path.join(args.checkpoint_dir, 'model', 'model_epoch{}.pth'.format(idx)) for idx in range(args.epoch)]
        for resume_path in resume_paths:
          assert(os.path.exists(resume_path)), resume_path
      else:
        resume_paths = os.path.join(args.model_dir, 'model_epoch{}.pth'.format(args.test_epoch))
        resume_paths = [resume_paths]
    else:
      resume_paths = [args.resume_path]

    duration = 0
    print('Start of testing ...')

    for resume_path in resume_paths:
      if os.path.exists(resume_path):
        print('Model: {}'.format(resume_path))
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'])
        model = model.cuda() if args.enable_cuda else model
        model = nn.DataParallel(model) if ((args.batch > 1) and args.enable_cuda) else model  # parallel after resuming
        args.current_epoch = checkpoint['epoch']

        with torch.no_grad():
          time_start = time.time()

          if not args.enable_viz:
            valid(model, valid_dataloader, writer, valid_set.names, args)  # testset has ground truth so use valid()
          else:
            test(model, valid_dataloader, args)

          duration += time.time() - time_start

        torch.cuda.empty_cache()
        print('====> Resume epoch: {}, valid avg loss: {:.4f}' \
              .format(args.current_epoch, checkpoint['loss']))

    print('End of testing, avg test time for each model {:.4f} h' \
          .format(duration / len(resume_paths) / 3600))