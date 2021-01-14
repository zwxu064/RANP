import torch
import torch.nn as nn
import os
from third_party.PSM.models import *
from third_party.efficient_3DCNN.model import generate_model
from third_party.unet.model import UNet3D
from third_party.thop.thop.profile import profile
from pruning.pytorch_snip.prune import pruning
from configs import set_config
from aux.utils import weight_init


if __name__ == "__main__":
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.enabled = True

  min_sparsity, max_sparsity = 0., 1.
  opt = set_config()
  opt.network_name = opt.model

  if opt.dataset == 'ucf101':
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
  elif opt.dataset == 'brats':
    opt.spatial_size = 128
    opt.year = 2018
    opt.prune_spatial_size = 96
    opt.num_of_fmaps = 4
    opt.n_class = 5
    opt.enable_deepmodel_pooling = True
    opt.data_dir = 'datasets/BraTS'
  elif opt.dataset == 'shapenet':
    opt.spatial_size = 128
    opt.dim = 64
    opt.scale = 1
    opt.prune_spatial_size = 64
    opt.num_of_fmaps = 4
    opt.n_class = 50
    opt.data_dir = 'datasets/ShapeNet'
  elif opt.dataset == 'sceneflow':
    opt.data_dir = 'datasets/SceneFlow'
  else:
    assert False

  # Set grad file, not generate here to avoid loading dataloader, loss things.
  # Run from the main file, that is train_ucf101.py
  grad_mode = 'raw' if opt.enable_raw_grad else 'abs'
  in_channels = 3

  if opt.dataset == 'ucf101':
    if opt.network_name == 'mobilenetv2':  # was in rebuttal previously
      file_path = 'data/ucf101/ucf101_{}_sz{}_{}_{}.npy'.format(
        opt.network_name, opt.prune_spatial_size, opt.weight_init, grad_mode)
    elif opt.network_name == 'i3d':
      file_path = 'data/ucf101/ucf101_{}_sz{}_{}_{}.npy'.format(
        opt.network_name, opt.prune_spatial_size, opt.weight_init, grad_mode)

    model, _ = generate_model(opt)
  elif opt.dataset == 'brats':
    in_channels, out_channels, final_sigmoid, f_maps = 4, opt.n_class, False, 32

    if opt.weight_init == 'xn':
      file_path = 'data/brats/brats2018_kernel_hidden_prune_grad_sz{}_dim{}_'\
                  'scale{}_fmap{}_depth{}_width{}_{}.npy'. \
        format(opt.prune_spatial_size, opt.prune_spatial_size,
               1, f_maps, opt.number_of_fmaps, opt.width, grad_mode)
    else:
      file_path = 'data/brats/brats2018_kernel_hidden_prune_grad_sz{}_dim{}_'\
                  'scale{}_fmap{}_depth{}_width{}_init{}_{}.npy'. \
        format(opt.prune_spatial_size, opt.prune_spatial_size, 1, f_maps,
               opt.number_of_fmaps, opt.width, opt.weight_init, grad_mode)

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
  elif opt.dataset == 'shapenet':
    in_channels, out_channels, final_sigmoid, f_maps = 1, opt.n_class, False, 64

    if opt.weight_init == 'xn':
      file_path = 'data/shapenet/shapenet_kernel_hidden_prune_grad_sz{}_'\
                  'dim{}_scale{}_fmap{}_depth{}_width{}_{}.npy'. \
        format(opt.prune_spatial_size, opt.prune_spatial_size,
               1, f_maps, opt.number_of_fmaps, opt.width, grad_mode)
    else:
      file_path = 'data/shapenet/shapenet_kernel_hidden_prune_grad_sz{}_'\
                  'dim{}_scale{}_fmap{}_depth{}_width{}_init{}_{}.npy'. \
        format(opt.prune_spatial_size, opt.prune_spatial_size,
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
  elif opt.dataset == 'sceneflow':
    file_path = 'data/stereo/stereo_kernel_hidden_prune_grad_abs.npy'
    model = stackhourglass(opt.maxdisp)
  else:
    file_path = None
    model = None

  assert (model is not None) and (file_path is not None) \
         and os.path.exists(file_path), file_path

  weight_init(model, mode=opt.weight_init)
  model = model.cuda()

  # Get resource list
  if opt.dataset == 'brats':
    profile_input = torch.randn(1, 4, opt.spatial_size, opt.spatial_size,
                                opt.spatial_size).cuda()
  elif opt.dataset == 'shapenet':
    profile_input = torch.randn(1, 1, opt.prune_spatial_size,
                                opt.prune_spatial_size,
                                opt.prune_spatial_size).cuda()
  elif opt.dataset == 'ucf101':  # the batch is 8 not 1 for mobilenetv2 in the rebuttal
    profile_input = torch.randn(1, in_channels, opt.sample_duration,
                                opt.sample_size,
                                opt.sample_size).cuda()
  elif opt.dataset == 'sceneflow':
    profile_input_L = torch.randn(3, 3, 256, 512).cuda()
    profile_input_R = torch.randn(3, 3, 256, 512).cuda()

  if opt.dataset == 'sceneflow':
    flops_full, params_full, memory_full, resource_list = profile(
      model, inputs=(profile_input_L, profile_input_R), verbose=False, resource_list_type=opt.resource_list_type)
  else:
    flops_full, params_full, memory_full, resource_list = profile(
      model, inputs=(profile_input,), verbose=False, resource_list_type=opt.resource_list_type)

  optimal_sparsity = 0.

  # Loop until finding the optimal max sparsity
  while (1):
    print('Loop:', min_sparsity, max_sparsity)
    model, train_loader, criterion = None, None, None

    if opt.dataset == 'ucf101':
      enable_3dunet = False
      network_name = opt.network_name
      width = 0
    elif opt.dataset == 'sceneflow':
      enable_3dunet = False
      network_name = 'psm'
      width = 0
    else:
      enable_3dunet = True
      network_name = '3dunet'
      width = opt.width

    sparsity = (min_sparsity + max_sparsity) / 2

    if opt.enable_param_prune:
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
