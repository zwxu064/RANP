import torch
import torch.nn as nn
import os
import copy
from third_party.efficient_3DCNN.opts import parse_opts as set_ucf101_config
from third_party.efficient_3DCNN.model import generate_model
from third_party.unet.model import UNet3D
from third_party.thop.thop.profile import profile
from pruning.pytorch_snip.prune import pruning
from train_shapenet import set_config as set_shapenet_config
from train_brats import set_config as set_brats_config
from utils import weight_init


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES']="0"  # TODO can only use 0, check later
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.enabled = True

  min_sparsity, max_sparsity = 0., 1.
  dataset = 'ucf101'

  if dataset == 'ucf101':
    opt = set_ucf101_config()
    opt.model = 'i3d'
    opt.network_name = opt.model

    if opt.network_name == 'i3d':
      opt.sample_duration = 16
      opt.sample_size = 224
      opt.prune_spatial_size = opt.sample_size
    elif opt.network_name == 'mobilenetv2':
      opt.sample_duration = 16
      opt.sample_size = 112
      opt.groups = 3
      opt.width_mult = 1.0
      opt.prune_spatial_size = opt.sample_size
    else:
      assert False
  elif dataset == 'brats':
    opt = set_brats_config()
    opt.spatial_size = 128
    opt.year = 2018
    opt.prune_spatial_size = 96
    opt.num_of_fmaps = 4
    opt.n_class = 5
    opt.enable_deepmodel_pooling = True
    opt.data_dir = 'datasets/{}/BraTS'.format(opt.server)
  elif dataset == 'shapenet':
    opt = set_shapenet_config()
    opt.spatial_size = 128
    opt.dim = 64
    opt.scale = 1
    opt.prune_spatial_size = 64
    opt.num_of_fmaps = 4
    opt.n_class = 50
    opt.data_dir = 'datasets/{}/ShapeNet'.format(opt.server)
  else:
    assert False

  enable_param_prune = False
  opt.acc_mode = 'sum'
  opt.enable_raw_grad = False  # abs+sum MPMG-sum
  opt.resource_list_type = 'vanilla'
  opt.resource_list_lambda = 80
  opt.weight_init = "xn"

  # Set grad file, not generate here to avoid loading dataloader, loss things.
  # Run from the main file, that is train_ucf101.py
  grad_mode = 'raw' if opt.enable_raw_grad else 'abs'
  in_channels = 3

  if dataset == 'ucf101':
    if opt.network_name == 'mobilenetv2':  # was in rebuttal previously
      file_path = 'data/{}/3DV/ucf101_{}_sz{}_{}_{}.npy'.format(
        opt.server, opt.network_name, opt.prune_spatial_size, opt.weight_init,
        grad_mode)
      model, _ = generate_model(opt)
    elif opt.network_name == 'i3d':
      file_path = 'data/{}/3DV/ucf101_{}_sz{}_{}_{}.npy'.format(
        opt.server, opt.network_name, opt.prune_spatial_size, opt.weight_init,
        grad_mode)

    model, _ = generate_model(opt)
  elif dataset == 'brats':
    in_channels, out_channels, final_sigmoid, f_maps = 4, opt.n_class, False, 32

    if opt.weight_init == 'xn':
      file_path = 'data/{}/main_paper/brats2018_kernel_hidden_prune_grad_sz{}_dim{}_'\
                  'scale{}_fmap{}_depth{}_width{}_{}.npy'. \
        format(opt.server, opt.prune_spatial_size, opt.prune_spatial_size,
               1, f_maps, opt.number_of_fmaps, opt.width, grad_mode)
    else:
      file_path = 'data/{}/main_paper/brats2018_kernel_hidden_prune_grad_sz{}_dim{}_'\
                  'scale{}_fmap{}_depth{}_width{}_init{}_{}.npy'. \
        format(opt.server, opt.prune_spatial_size, opt.prune_spatial_size,
               1, f_maps, opt.number_of_fmaps, opt.width, opt.weight_init,
               grad_mode)

    model = UNet3D(in_channels,
                   out_channels,
                   final_sigmoid,
                   f_maps=f_maps,
                   layer_order='cbr',
                   num_groups=4,
                   enable_prob=False,
                   number_of_fmaps=opt.number_of_fmaps,
                   enable_deepmodel_pooling=opt.enable_deepmodel_pooling,
                   width=opt.width,
                   res_type=opt.res_type)
  elif dataset == 'shapenet':
    in_channels, out_channels, final_sigmoid, f_maps = 1, opt.n_class, False, 64

    if opt.weight_init == 'xn':
      file_path = 'data/{}/main_paper/shapenet_kernel_hidden_prune_grad_sz{}_'\
                  'dim{}_scale{}_fmap{}_depth{}_width{}_{}.npy'. \
        format(opt.server, opt.prune_spatial_size, opt.prune_spatial_size,
               1, f_maps, opt.number_of_fmaps, opt.width, grad_mode)
    else:
      file_path = 'data/{}/main_paper/shapenet_kernel_hidden_prune_grad_sz{}_'\
                  'dim{}_scale{}_fmap{}_depth{}_width{}_init{}_{}.npy'. \
        format(opt.server, opt.prune_spatial_size, opt.prune_spatial_size,
               1, f_maps, opt.number_of_fmaps, opt.width, opt.weight_init,
               grad_mode)

    model = UNet3D(in_channels,
                   out_channels,
                   final_sigmoid,
                   f_maps=f_maps,
                   layer_order='cbr',
                   num_groups=8,
                   number_of_fmaps = opt.number_of_fmaps,
                   enable_deepmodel_pooling=opt.enable_deepmodel_pooling,
                   width=opt.width,
                   res_type=opt.res_type)

  else:
    file_path = None
    model = None

  assert (model is not None) and (file_path is not None) \
         and os.path.exists(file_path), file_path

  weight_init(model, mode=opt.weight_init)

  model = model.cuda()
  model = nn.DataParallel(model, device_ids=None)

  # Get resource list
  if dataset == 'brats':
    profile_input = torch.randn(1, 4, opt.spatial_size, opt.spatial_size,
                                opt.spatial_size).cuda()
  elif dataset == 'shapenet':
    profile_input = torch.randn(1, 1, opt.prune_spatial_size,
                                opt.prune_spatial_size,
                                opt.prune_spatial_size).cuda()
  elif dataset == 'ucf101':  # the batch is 8 not 1 for mobilenetv2 in the rebuttal
    profile_input = torch.randn(1, in_channels, opt.sample_duration,
                                opt.sample_size,
                                opt.sample_size).cuda()

  flops_full, params_full, memory_full, resource_list = profile(
    model, inputs=(profile_input,), verbose=False, resource_list_type=opt.resource_list_type)
  optimal_sparsity = 0.

  # Loop until finding the optimal max sparsity
  while (1):
    print('Loop:', min_sparsity, max_sparsity)
    model, train_loader, criterion = None, None, None

    if dataset == 'ucf101':
      enable_3dunet = False
      network_name = opt.network_name
      width = 0
    else:
      enable_3dunet = True
      network_name = '3dunet'
      width = opt.width

    sparsity = (min_sparsity + max_sparsity) / 2

    if enable_param_prune:
      opt.param_sparsity = sparsity
      opt.enable_param_prune = True
    else:
      opt.neuron_sparsity = sparsity
      opt.enable_neuron_prune = True

    outputs = pruning(file_path, model, train_loader, criterion, opt,
                      enable_3dunet=enable_3dunet, enable_hidden_sum=False, width=width,
                      resource_list=resource_list, network_name=network_name)

    if outputs[0] == 0:
      min_sparsity = sparsity
      optimal_sparsity = sparsity
      print('Fine and continue:', optimal_sparsity)
    else:
      max_sparsity = sparsity

    if (max_sparsity - min_sparsity) <= 1e-4:
      break

  print('Full model, flops: {:.4f}G, params: {:.4f}MB, memory: {:.4f}MB' \
        .format(flops_full / 1e9, params_full * 4 / (1024 ** 2), memory_full * 4 / (1024 ** 2)))
  print('Final optimal sparsity:', optimal_sparsity)
