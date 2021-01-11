import os, torch, time, math, copy
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from pruning.pytorch_snip.prune import pruning, apply_hidden_mask, do_statistics_model, dump_neuron_per_layer
from pruning_related import refine_model
from third_party.unet.model import UNet3D, UNet3DSSC
from third_party.thop.thop.profile import profile
from aux.utils import AverageMeter, load_optimizer_state_dict, check_dir, write_to_tensorboard
from aux.utils import weight_init, get_model_state_dict, model_transfer_last_layer
from aux.viz_voxel import viz_voxel
from aux.fully_convolutional import store_voxel, iou
from aux.dataloader import SHAPENET
from configs import set_config


def train(model, dataset, optimizer, writer, args):
  model.train()
  status = {}
  train_loss = AverageMeter()
  duration, counts = 0, 0
  dataset_len = len(dataset)

  for param_group in optimizer.param_groups:
    param_group['lr'] = args.lr * math.exp((1 - args.current_epoch) * args.lr_decay)

  time_start = time.time()
  current_iter = args.current_epoch * dataset_len

  for idx, data in enumerate(dataset):
    input, gt = data['x'], data['y']
    actual_batch = input.size(0)

    if args.enable_cuda:
      input, gt = input.cuda(), gt.cuda()

    optimizer.zero_grad()

    duration_start = time.time()
    prediction = model(input)
    duration_per = time.time() - duration_start

    if args.enable_ssc_unet:
      if idx < 100:
        counts += 1
        duration += duration_per
      else:
        break

    prediction = prediction.reshape(actual_batch, args.n_class, -1)
    gt = gt.reshape(actual_batch, -1)
    loss = criterion(prediction, gt)
    store_voxel(status, data, prediction, gt)
    loss.backward()
    optimizer.step()

    train_loss.update(loss.item(), actual_batch)
    print('Epoch: {}, batch: {}/{}, train loss: {:.4f}'.format(args.current_epoch, idx, dataset_len, train_loss.avg))

    current_iter += 1
    train_iter_result = {'mode': 'train', 'batch_iter': current_iter, 'train_loss_iter': train_loss.avg}
    write_to_tensorboard('scalar', writer, train_iter_result)

  if args.enable_ssc_unet:
    print('=====> Avg time: {}s over {} loops.'.format(duration / counts, counts))

  duration = time.time() - time_start
  iou_value = iou(status, class_first=True)
  print('Epoch: {}, train loss: {:.4f}, IoU: {:.4f}, time: {:.4f}s' \
        .format(args.current_epoch, train_loss.avg, iou_value['iou'], duration))
  train_result = {'mode': 'train', 'epoch': args.current_epoch, 'train_loss': train_loss.avg}
  write_to_tensorboard('scalar', writer, train_result)


def valid(model, dataset, writer, args):
  model.eval()
  status = {}
  valid_loss = AverageMeter()
  time_start = time.time()
  dataset_len = len(dataset)

  for idx, data in enumerate(dataset):
    input, gt = data['x'], data['y']
    actual_batch = input.size(0)

    if args.enable_cuda:
      input, gt = input.cuda(), gt.cuda()

    prediction = model(input)
    prediction = prediction.reshape(actual_batch, args.n_class, -1)
    gt = gt.reshape(actual_batch, -1)
    loss = criterion(prediction, gt)
    valid_loss.update(loss.item(), actual_batch)
    store_voxel(status, data, prediction, gt)
    print('Epoch: {}, batch: {}/{}, valid loss: {:.4f}'.format(args.current_epoch, idx, dataset_len, valid_loss.avg))

  duration = time.time() - time_start
  iou_value = iou(status, class_first=True)
  print('Epoch: {}, valid loss: {:.4f}, IoU: {:.4f}, time: {:.4f}s' \
        .format(args.current_epoch, valid_loss.avg, iou_value['iou'], duration))
  valid_result = {'mode': 'valid', 'epoch': args.current_epoch, 'valid_loss': valid_loss.avg,
                  'mean_iou': iou_value['iou']}
  write_to_tensorboard('scalar', writer, valid_result)

  return iou_value['iou'], valid_loss.avg


def test(model, dataset, args):
  model.eval()

  for idx, data in enumerate(dataset):
    time_start = time.time()
    input, gt = data['x'], data['y']
    file_path = data['file_path'][0]
    class_offset = data['class_offset'][0]
    num_class = data['num_class'][0]
    name = file_path.split('/')[-1]
    name = name.split('.')[0]

    if args.enable_cuda:
      input, gt = input.cuda(), gt.cuda()

    prediction = model(input)
    mask = (input[0, 0].detach().cpu().numpy() == 1)
    voxel = prediction[0][class_offset : class_offset + num_class].argmax(0).detach().cpu().numpy()
    voxel = np.reshape(voxel, (args.spatial_size, args.spatial_size, args.spatial_size))
    duration = time.time() - time_start
    print('Test, batch: {}, time: {:.4f}s'.format(idx, duration))
    enable_save = False
    close_time = 1

    if args.enable_viz:
      if idx != 1: continue
      viz_voxel(voxel=voxel, mask=mask, enable_close_time=close_time, data_root='viz_figures', enable_save=enable_save,
                title='input_{}_left'.format(name), elevation=30, azimuth=-45, fixed_color='white')
      viz_voxel(voxel=voxel, mask=mask, enable_close_time=close_time, data_root='viz_figures', enable_save=enable_save,
                title='input_{}_right'.format(name), elevation=30, azimuth=45, fixed_color='white')
      viz_voxel(voxel=voxel, mask=mask, enable_close_time=close_time, data_root='viz_figures', enable_save=enable_save,
                title='prediction_{}_left'.format(name), elevation=30, azimuth=-45)
      viz_voxel(voxel=voxel, mask=mask, enable_close_time=close_time, data_root='viz_figures', enable_save=enable_save,
                title='prediction_{}_right'.format(name), elevation=30, azimuth=45)
      viz_voxel(voxel=gt.squeeze().cpu().numpy(), mask=mask, data_root='viz_figures', enable_save=enable_save,
                enable_close_time=close_time, title='GT_{}_left'.format(name), elevation=30, azimuth=-45)
      viz_voxel(voxel=gt.squeeze().cpu().numpy(), mask=mask, data_root='viz_figures', enable_save=enable_save,
                enable_close_time=0, title='GT_{}_right'.format(name), elevation=30, azimuth=45)


if __name__ == '__main__':
  # Set env
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.enabled = True

  # Set seed
  seed = 2019
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  # Config
  args = set_config()
  enable_dump_neuron_per_layer = False

  # should be fa
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
    model = model_transfer_last_layer(model, 1, args.n_class, args.weight_init)
  else:
    in_channels, out_channels, final_sigmoid, f_maps = 1, args.n_class, False, 64  # f_map=64

    if args.enable_ssc_unet:
      model = UNet3DSSC()
    else:
      model = UNet3D(in_channels,
                     out_channels,
                     final_sigmoid,
                     f_maps=f_maps,
                     layer_order='cbr',
                     num_groups=8,
                     number_of_fmaps = args.number_of_fmaps,
                     enable_deepmodel_pooling=args.enable_deepmodel_pooling,
                     width=args.width,
                     res_type=args.res_type)
    weight_init(model, mode=args.weight_init)

  # ==== Optimizer
  if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
  else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.w_decay, nesterov=True)

  # Components from SSC
  criterion = nn.CrossEntropyLoss(ignore_index=255)  # ignore empty space marked as 255

  if args.enable_cuda:
    model = model.cuda()

  # Tensorboard
  writer = SummaryWriter(args.event_dir)

  # Datasets
  # 3DCNN for sparse point cloud is not very good when it is too sparse, using hard padding
  if args.enable_hard_padding:
    assert args.spatial_size > 96, 'Error! Hard padding only works for>96 spatial size due to too sparse input'
    dataset_dim = 64
    dataset_scale = args.spatial_size / dataset_dim
  else:
    dataset_dim = args.spatial_size
    dataset_scale = 1

  print('For dataset: spatial size: {}, dim: {}, scale: {}'.format(args.spatial_size, dataset_dim, dataset_scale))

  trainset = SHAPENET('train', args.spatial_size, dataset_dim, dataset_scale, args.data_dir,
                      enable_random_trans=False, enable_random_rotate=False, enable_voxel_gt=True,
                      enable_hard_padding=args.enable_hard_padding)
  validset = SHAPENET('valid', args.valid_spatial_size, dataset_dim, dataset_scale, args.data_dir, enable_voxel_gt=True,
                      enable_hard_padding=args.enable_hard_padding, target_class=args.test_target_class)
  testset = SHAPENET('test', args.valid_spatial_size, dataset_dim, dataset_scale, args.data_dir,
                     enable_random_trans=False, enable_random_rotate=False, enable_voxel_gt=True,
                     enable_hard_padding=args.enable_hard_padding)
  train_dataloader = DataLoader(trainset, batch_size=args.batch, num_workers=args.batch * 4, shuffle=True)
  valid_dataloader = DataLoader(validset, batch_size=1, num_workers=4, shuffle=False)
  test_dataloader = DataLoader(testset, batch_size=1, num_workers=4, shuffle=False)

  # Calculate Profile of Full Model
  model_full = copy.deepcopy(model)
  profile_input = torch.randn(1, 1, args.spatial_size, args.spatial_size, args.spatial_size).cuda()
  flops_full, params_full, memory_full, resource_list = profile(model_full.cuda(), inputs=(profile_input,), verbose=False,
                                                                resource_list_type=args.resource_list_type)
  del model_full

  # Prune including kernels and hidden layers
  if args.enable_neuron_prune or args.enable_hidden_layer_prune or args.enable_param_prune:
    spatial_size_org = args.spatial_size
    dim_org = args.dim
    args.spatial_size = args.prune_spatial_size
    args.dim = args.prune_spatial_size
    grad_mode = 'raw' if args.enable_raw_grad else 'abs'

    if args.weight_init == 'xn':
      file_path = 'data/shapenet/shapenet_kernel_hidden_prune_grad_sz{}_dim{}_scale{}_'\
                  'fmap{}_depth{}_width{}_{}.npy'. \
        format(args.prune_spatial_size, args.prune_spatial_size, 1, f_maps,
               args.number_of_fmaps, args.width, grad_mode)
    else:
      file_path = 'data/shapenet/shapenet_kernel_hidden_prune_grad_sz{}_dim{}_scale{}_'\
                  'fmap{}_depth{}_width{}_init{}_{}.npy'. \
        format(args.prune_spatial_size, args.prune_spatial_size, 1, f_maps,
               args.number_of_fmaps, args.width,
               args.weight_init, grad_mode)

    assert (args.batch == 1 if (not os.path.exists(file_path)) else True)

    outputs = pruning(file_path, model, train_dataloader, criterion, args,
                      enable_3dunet=True, enable_hidden_sum=enable_hidden_sum,
                      width=args.width, resource_list=resource_list)
    assert outputs[0] == 0
    neuron_mask_clean, hidden_mask = outputs[1], outputs[2]

    if args.enable_neuron_prune or args.enable_param_prune:
      n_params_org, n_neurons_org = do_statistics_model(model)
      new_model = refine_model(model, neuron_mask_clean, enable_3dunet=True, width=args.width)

      if enable_dump_neuron_per_layer:
        dump_neuron_per_layer(copy.deepcopy(model), copy.deepcopy(new_model))

      del model
      model = new_model.cpu()
      weight_init(model, mode=args.weight_init)
      model = model.cuda()
      n_params_refined, n_neurons_refined = do_statistics_model(model)
      print('Statistics, org, params: {}, neurons: {}; refined, '\
            'params: {} ({:.4f}%), neurons: {} ({:.4f}%)' \
            .format(n_params_org,
                    n_neurons_org,
                    n_params_refined,
                    n_params_refined * 100 / n_params_org,
                    n_neurons_refined,
                    n_neurons_refined * 100 / n_neurons_org))

      # For new model
      if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
      else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay, nesterov=True)

    if args.enable_hidden_layer_prune:
      apply_hidden_mask(model, hidden_mask, enable_hidden_sum=enable_hidden_sum)

    args.spatial_size = spatial_size_org
    args.dim = dim_org

  # Resume
  if (args.model_dir is not None) and (args.resume_epoch >= 0):
    resume_path = os.path.join(args.model_dir, 'model_epoch{}.pth'.format(args.resume_epoch))

    if os.path.exists(resume_path):
      print('Resume from epoch: {}'.format(args.resume_epoch))
      checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
      model.load_state_dict(checkpoint['model'])
      # optimizer from parallel model
      optimizer = load_optimizer_state_dict(checkpoint['optimizer'], optimizer, enable_cuda=args.enable_cuda)
    else:
      print('Resume epoch: {} failed, start from 0'.format(args.resume_epoch))
      args.resume_epoch = -1

  # Calculate Flops, Params, Memory of New Model
  model_flops = copy.deepcopy(model)
  flops, params, memory, _ = profile(model_flops.cuda(), inputs=(profile_input,), verbose=False,
                                     enable_layer_neuron_display=args.enable_layer_neuron_display)
  print('Full model, flops: {:.4f}G, params: {:.4f}MB, memory: {:.4f}MB' \
        .format(flops_full / 1e9, params_full * 4 / (1024 ** 2), memory_full * 4 / (1024 ** 2)))
  print('New model, flops: {:.4f}G, params: {:.4f}MB, memory: {:.4f}MB' \
        .format(flops / 1e9, params * 4 / (1024 ** 2), memory * 4 / (1024 ** 2)))
  del model_flops, profile_input

  print(args)
  # print(model)

  if (args.save_transfer_model_path != '') and (not os.path.exists(args.save_transfer_model_path)):
    print('====> Save transfer model to {}'.format(args.save_transfer_model_path))
    torch.save(model.cpu(), args.save_transfer_model_path)
    exit()

  # Train and valid
  if args.enable_train:
    if True:
      model = nn.DataParallel(model) if ((args.batch > 1) and args.enable_cuda) else model
    else:
      torch.distributed.init_process_group(backend="nccl", init_method="env://")
      model = nn.parallel.DistributedDataParallel(model) if ((args.batch > 1) and args.enable_cuda) else model

    best_iou = 0
    train_duration, valid_duration = 0, 0
    print('Start of training ...')

    for epoch in range(args.resume_epoch+1, args.epoch):
      if epoch > 105: continue
      args.current_epoch = epoch
      time_start = time.time()
      train(model, train_dataloader, optimizer, writer, args)
      train_duration += time.time() - time_start
      torch.cuda.empty_cache()

      # if (args.spatial_size <= 96) or ((args.spatial_size > 96) and (epoch >= args.epoch - 10)):
      if epoch >= args.valid_min_epoch:
        with torch.no_grad():
          time_start = time.time()
          mean_iou, valid_loss = valid(model, valid_dataloader, writer, args)
          valid_duration += time.time() - time_start
          torch.cuda.empty_cache()

          if args.model_dir is not None:
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

          if args.enable_viz:
            mean_iou, test_loss = None, None
            test(model, valid_dataloader, args)
          else:
            mean_iou, test_loss = valid(model, valid_dataloader, writer, args)  # testset has ground truth so use valid() # TODO replace validset by testset

          duration += time.time() - time_start

        torch.cuda.empty_cache()
        print('====> Resume epoch: {}, valid avg loss: {:.4f}; test avg loss: {:.4f}, mean IoU: {:.4f}' \
              .format(args.current_epoch, checkpoint['loss'], test_loss, mean_iou))

    print('End of testing, avg test time for each model {:.4f} h' \
          .format(duration / len(resume_paths) / 3600))
