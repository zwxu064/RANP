import os, random, torch, time, math, copy, glob
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data, torch.nn.parallel
from third_party.PSM.dataloader import listflowfile as lt
from third_party.PSM.dataloader import SecenFlowLoader as DA
from third_party.PSM.models import *
from configs import set_config
from third_party.thop.thop.profile import profile
from pruning.pytorch_snip.prune import pruning, do_statistics_model, dump_neuron_per_layer
from pruning_related import refine_model_PSM
from aux.utils import weight_init, AverageMeter


def cal_acc(disp_est, disp_gt, accuracies, max_disp=192, mask=None, dataset=None):
  if dataset.find('KITTI') > -1:
    if mask is None:
      mask = (disp_gt > 0).float()
    else:
      mask = ((disp_gt > 0) & (mask == 1)).float()
  else:
    if mask is None:
      mask = ((disp_gt >= 0) & (disp_gt < max_disp)).float()
    else:
      mask = ((disp_gt >= 0) & (disp_gt < max_disp) & (mask == 1)).float()

  threshold_vector = [1., 2., 3., 5.]
  diff = torch.abs(disp_est.cpu() - disp_gt.cpu())

  if dataset in ['KITTI2015', 'KITTI-single']:
    per_threshold = 0.05
  else:
    per_threshold = 1.0

  assert (len(threshold_vector) + 1 == len(accuracies))
  current_accuracies = [-1, -1, -1, -1, -1]
  disp_est_size = disp_est.size()
  batch = disp_est_size[0]

  if len(disp_est_size) == 3:
    disp_gt = disp_gt.unsqueeze(1)
    mask = mask.unsqueeze(1)
    diff = diff.unsqueeze(1)
    disp_num = 1
  else:
    disp_num = disp_est.size(1)

  # ==== Accuracies
  for i in range(len(accuracies) - 1):
    valid_area = ((diff <= threshold_vector[i]) & (diff <= (per_threshold * disp_gt))).float() * mask

    # 20190927 one image by one image for average
    for batch_ind in range(batch):
      for disp_ind in range(disp_num):  # left and/or right
        acc = valid_area[batch_ind, disp_ind].double().sum() / mask[batch_ind, disp_ind].double().sum()
        current_accuracies[i] = acc.data.cpu().numpy().item()
        accuracies[i].update(current_accuracies[i])

  # ==== EPE
  for batch_ind in range(batch):
    for disp_ind in range(disp_num):  # left and/or right
      epe = (diff[batch_ind, disp_ind] * mask[batch_ind, disp_ind]).double().sum() / \
            mask[batch_ind, disp_ind].double().sum()
      current_accuracies[4] = epe.data.cpu().numpy().item()
      accuracies[4].update(current_accuracies[4])

  return accuracies, current_accuracies


def display_model_structure(model):
  idx = 0
  for key, layer in model.named_modules():
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv3d,
                          nn.ConvTranspose2d, nn.ConvTranspose3d)):
      idx += 1


def get_model_state_dict(model):
  if hasattr(model, 'module'):
    model_state_dict = model.module.state_dict()
  else:
    model_state_dict = model.state_dict()

  return model_state_dict


def load_optimizer_state_dict(checkpoint, optimizer, enable_cuda=True):
  optimizer.load_state_dict(checkpoint)

  for state in optimizer.state.values():
    for k, v in state.items():
      if isinstance(v, torch.Tensor):
        if enable_cuda:
          state[k] = v.cuda()
        else:
          state[k] = v.cpu()

  return optimizer


def train(imgL, imgR, disp_L, model, optimizer):
  model.train()
  imgL = Variable(torch.FloatTensor(imgL))
  imgR = Variable(torch.FloatTensor(imgR))
  disp_L = Variable(torch.FloatTensor(disp_L))

  if args.cuda:
    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

  mask = disp_true < args.maxdisp
  mask.detach_()
  optimizer.zero_grad()

  output1, output2, output3 = model(imgL, imgR)
  output1 = torch.squeeze(output1, 1)
  output2 = torch.squeeze(output2, 1)
  output3 = torch.squeeze(output3, 1)
  loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], reduction='mean') \
         + 0.7 * F.smooth_l1_loss(output2[mask], disp_true[mask], reduction='mean') \
         + F.smooth_l1_loss(output3[mask], disp_true[mask], reduction='mean')

  loss.backward()
  optimizer.step()

  return loss


def test(imgL, imgR, disp_true, model):
  model.eval()
  imgL = Variable(torch.FloatTensor(imgL))
  imgR = Variable(torch.FloatTensor(imgR))

  if args.cuda:
    imgL, imgR = imgL.cuda(), imgR.cuda()

  mask = disp_true < 192

  with torch.no_grad():
    output3, _ = model(imgL, imgR)

  output = torch.squeeze(output3.data.cpu(), 1)[:, 4 :, :]

  if len(disp_true[mask])==0:
    loss = 0
  else:
    loss = torch.mean(torch.abs(output[mask]-disp_true[mask]))  # end-point-error

  return loss, output3


# def adjust_learning_rate(optimizer, epoch):
#   lr = 0.001
#   print(lr)
#
#   for param_group in optimizer.param_groups:
#     param_group['lr'] = lr


def main(args):
  # Dataloader
  all_left_img, all_right_img, all_left_disp, test_left_img, \
    test_right_img, test_left_disp = lt.dataloader(args.datapath)

  # batch_size=12
  TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
    batch_size=args.batch, shuffle=True, num_workers=args.batch * 2, drop_last=False)

  # batch_size=8
  TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
    batch_size=8, shuffle=False, num_workers=8, drop_last=False)

  # Model
  model = stackhourglass(args.maxdisp)
  weight_init(model, mode=args.weight_init)

  # Optimizer
  optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

  # =========================================================
  # Calculate Profile of Full Model
  model_full = copy.deepcopy(model)
  profile_input_L = torch.randn(3, 3, 256, 512).cuda()
  profile_input_R = torch.randn(3, 3, 256, 512).cuda()

  flops_full, params_full, memory_full, resource_list = \
    profile(model_full.cuda(),
            inputs=(profile_input_L, profile_input_R),
            verbose=False,
            resource_list_type=args.resource_list_type)

  # display_model_structure(model_full)

  del model_full
  print('Full model, flops: {:.4f}G, params: {:.4f}MB, memory: {:.4f}MB' \
        .format(flops_full / 1e9, params_full * 4 / (1024 ** 2), memory_full * 4 / (1024 ** 2)))

  # Prune including kernels and hidden layers
  if args.enable_neuron_prune or args.enable_hidden_layer_prune or args.enable_param_prune:
    args.spatial_size = args.prune_spatial_size
    args.dim = args.prune_spatial_size
    grad_mode = 'raw' if args.enable_raw_grad else 'abs'

    if args.weight_init == 'xn':
      file_path = 'data/stereo/stereo_kernel_hidden_prune_grad_{}.npy'.format(grad_mode)
    else:
      file_path = 'data/stereo/stereo_kernel_hidden_prune_grad_init{}_{}.npy' \
        .format(args.weight_init, grad_mode)

    assert (args.batch == 1 if (not os.path.exists(file_path)) else True)

    outputs = pruning(file_path, model, TrainImgLoader, None, args,
                      enable_3dunet=False, enable_hidden_sum=False,
                      width=None, resource_list=resource_list, network_name='psm')
    assert outputs[0] == 0

    # neuron_mask_clean, hidden_mask = outputs[1], outputs[2]
    valid_neuron_list_clean, hidden_mask = outputs[1], outputs[2]

    if args.enable_neuron_prune or args.enable_param_prune:
      n_params_org, n_neurons_org = do_statistics_model(model)
      new_model = refine_model_PSM(model, valid_neuron_list_clean)

      if False:  # enable_dump_neuron_per_layer:
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

      # Calculate Flops, Params, Memory of New Model
      model_flops = copy.deepcopy(model)

      # display_model_structure(model_flops)

      flops, params, memory, _ = profile(model_flops.cuda(),
                                         inputs=(profile_input_L, profile_input_R),
                                         verbose=False,
                                         resource_list_type=args.resource_list_type)
      print('New model, flops: {:.4f}G, params: {:.4f}MB, memory: {:.4f}MB' \
            .format(flops / 1e9, params * 4 / (1024 ** 2), memory * 4 / (1024 ** 2)))
      del model_flops, profile_input_L, profile_input_R

      # For new model
      optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
  # =========================================================

  # print(model)
  print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

  # Resume
  if args.loadmodel is not None and os.path.isfile(args.loadmodel):
    state_dict = torch.load(args.loadmodel, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict['model'])
    optimizer = load_optimizer_state_dict(state_dict['optimizer'], optimizer, enable_cuda=args.cuda)

  if args.cuda:
    model.cuda()

  if args.enable_train:
    # Train
    start_full_time = time.time()
    epoch_start = int(args.loadmodel.split('.')[-2].split('_')[-1]) if (args.loadmodel is not None) else 0

    for epoch in range(epoch_start + 1, args.epochs + 1):
      print('This is %d-th epoch' % (epoch))
      torch.manual_seed(epoch)

      if args.cuda:
        torch.cuda.manual_seed(epoch)

      total_train_loss = 0
      # adjust_learning_rate(optimizer, epoch)

      for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
        start_time = time.time()
        loss = train(imgL_crop, imgR_crop, disp_crop_L, model, optimizer)
        print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
        total_train_loss += loss.item()

      print('Epoch %d total training loss = %.3f' %(epoch, total_train_loss / len(TrainImgLoader)))

      # SAVE
      if not os.path.exists(args.savemodel):
        os.makedirs(args.savemodel)

      savefilename = os.path.join(args.savemodel, 'checkpoint_' + str(epoch) + '.tar')
      model_state_dict = get_model_state_dict(model)
      torch.save({'epoch': epoch,
                  'model': model_state_dict,
                  'train_loss': total_train_loss / len(TrainImgLoader),
                  'optimizer': optimizer.state_dict()}, savefilename)

    print('Full training time = %.2f HR' %((time.time() - start_full_time)/3600))

    # Valid
    total_test_loss = 0
    valid_acc = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]

    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
      test_loss, output3 = test(imgL, imgR, disp_L, model)
      total_test_loss += test_loss.item()
      occ_mask = disp_L > 192
      valid_acc, _ = cal_acc(output3[:, 4:], disp_L, valid_acc, mask=occ_mask, max_disp=192, dataset=args.dataset)
      print('Iter %d test loss = %.3f' % (batch_idx, test_loss))

    print('Acc1: {:.4f}, acc2: {:.4f}, acc3: {:.4f}, acc5: {:.4f}, epe: {:.4f}; test loss: {:.4f}' \
          .format(valid_acc[0].avg, valid_acc[1].avg, valid_acc[2].avg, valid_acc[3].avg,
                  valid_acc[4].avg, total_test_loss / len(TestImgLoader)))
    # savefilename = os.path.join(args.savemodel, 'testinformation.tar')
    # torch.save({'test_loss': total_test_loss / len(TestImgLoader)}, savefilename)
  elif args.enable_test:
    if os.path.isdir(args.loadmodel):
      model_paths = glob.glob('{}/checkpoint_*.tar'.format(args.loadmodel))
    elif os.path.isfile(args.loadmodel):
      model_paths = [args.loadmodel]
    else:
      model_paths = []

    model_paths.sort(key=lambda x: int(x.split('_')[-1].split('.tar')[0]))

    for model_path in model_paths:
      assert os.path.exists(model_path)
      current_epoch = int(model_path.split('_')[-1].split('.tar')[0])
      state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
      model.load_state_dict(state_dict['model'])

      if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()

      total_test_loss = 0
      valid_acc = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]

      for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        test_loss, output3 = test(imgL, imgR, disp_L, model)
        total_test_loss += test_loss.item()
        occ_mask = disp_L > 192
        valid_acc, _ = cal_acc(output3[:, 4:], disp_L, valid_acc, mask=occ_mask, max_disp=192, dataset=args.dataset)
        print('Iter %d test loss = %.3f' % (batch_idx, test_loss))

      print('Epoch: {}, acc1: {:.4f}, acc2: {:.4f}, acc3: {:.4f}, acc5: {:.4f}, epe: {:.4f}; test loss: {:.4f}' \
            .format(current_epoch, valid_acc[0].avg, valid_acc[1].avg, valid_acc[2].avg, valid_acc[3].avg,
                    valid_acc[4].avg, total_test_loss / len(TestImgLoader)))


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

  args = set_config()
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  torch.manual_seed(args.seed)

  if args.cuda:
    torch.cuda.manual_seed(args.seed)

  main(args)
