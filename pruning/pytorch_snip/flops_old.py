import sys
import copy
import torch
import torch.nn as nn
import random
import numpy as np
sys.path.append('../flops')
sys.path.append('../../flops')
from thop.profile import profile
from ptflops.flops_counter import get_model_complexity_info
from torchvision.models import resnet50


def reparam_network(net, mask):
  net = copy.deepcopy(net)
  mask = copy.deepcopy(mask)
  prunable_layers = filter(lambda layer: isinstance(layer, nn.Conv2d)
                                         or isinstance(layer, nn.Linear)
                                         or isinstance(layer, nn.Conv3d)
                                         or isinstance(layer, nn.ConvTranspose3d),
                           net.modules())

  for idx, layer in enumerate(prunable_layers):
    weight_mask = mask[2 * idx]
    bias_mask = mask[2 * idx + 1]
    weight_mask = layer.weight.new_ones(layer.weight.shape) if (weight_mask is None) else weight_mask
    assert (layer.weight.shape == weight_mask.shape)
    layer.weight_mask = weight_mask  # create new variable "weight_mask" if it does not have it

    if bias_mask is not None:
      assert layer.bias.shape == bias_mask.shape
      layer.bias_mask = bias_mask  # create new variable "bias_mask" if it does not have it

  return net


def cal_flops(net, mask, input, enable_gflops=True, comment=''):
  net = copy.deepcopy(net)
  mask = copy.deepcopy(mask)
  input = [input] if (not isinstance(input, list)) else input
  net = reparam_network(net, mask)
  flops, params = profile(net, input, verbose=False)

  if enable_gflops:
    flops /= 10 ** 9

  return flops, params


if __name__ == '__main__':
  seed = 2019
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  # Test 1
  model = nn.Sequential(
    nn.Conv2d(3, 5, 2, stride=1, bias=True),
    nn.Conv2d(5, 1, 3, stride=1, bias=True),
    nn.Linear(1, 5, bias=True))

  mask = []
  mask.append(torch.randint(2, (5, 3, 2, 2)) * 0 + 1)
  mask.append(None)
  mask.append(torch.randint(2, (5, 1)) * 0 + 1)
  mask.append(torch.randint(2, (5,)) * 0 + 1)
  mask.append(None)
  mask.append(torch.randint(2, (5,)) * 0 + 1)

  batch = 1
  input = [torch.randn(batch, 3, 4, 4, dtype=torch.float32)]
  #
  # flops, params = cal_flops(model, mask, input, enable_gflops=False, comment='')
  # print('flops:', flops, ', params:', params)

  # Test 2
  # model = nn.Sequential(nn.Linear(10, 5, bias=True))
  # mask = []
  # mask.append(torch.randint(2, (5, 10)) * 0 + 1)
  # mask.append(torch.randint(2, (5,)) * 0 + 1)
  # batch = 1
  # input = [torch.randn(batch, 10, dtype=torch.float32)]
  #
  # flops, params = cal_flops(model, mask, input, enable_gflops=False, comment='')
  # print('flops:', flops, ', params:', params)

  # Test 3
  model = resnet50()
  input = torch.randn(1, 3, 224, 224)
  flops, params = profile(model, inputs=(input,))
  print('ResNet50 flops:{}, params:{}'.format(flops, params))

  # # The same above
  # flops, params = get_model_complexity_info(model, (in_c, h, w), as_strings=True, print_per_layer_stat=False, units='E')
  # print(flops, params)