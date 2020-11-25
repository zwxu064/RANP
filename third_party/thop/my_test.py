import sys
import copy
import torch
import torch.nn as nn
import random
import numpy as np
from thop.profile import profile
from torchvision.models import resnet50


if __name__ == '__main__':
  seed = 2019
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  model = nn.Sequential(
    nn.Conv2d(3, 5, 2, stride=1, bias=True),
    nn.Conv2d(5, 1, 3, stride=1, bias=True),
    nn.Linear(1, 5, bias=True))

  model = resnet50()
  input = torch.randn(1, 3, 224, 224)
  flops, params, memory = profile(model, inputs=(input,), verbose=False)
  print('ResNet50 flops:{}, params:{}, memory:{}'.format(flops, params, memory))
