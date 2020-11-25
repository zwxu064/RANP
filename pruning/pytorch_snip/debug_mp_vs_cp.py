import os
import torch
import torch.nn as nn
import copy
from neuron_prune import channel_prune, remove_redundant, do_statistics
from mp_prune import message_passing_prune
from utils import check_same
from network import VGG, LeNet_5_Caffe


seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

enable_bias = True
channel_sparsity = 0.9
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = VGG('C', num_classes=10, enable_bias=enable_bias).to(device)
model = LeNet_5_Caffe(enable_bias=enable_bias).to(device)

grads = []

for idx, layer in enumerate(model.modules()):
  if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    grads.append(torch.randn(layer.weight.size(), dtype=torch.float32, device=device).abs())

if enable_bias:
  for idx, layer in enumerate(model.modules()):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
      grads.append(torch.randn(layer.bias.size(), dtype=torch.float32, device=device).abs())

grads_flatten = torch.cat([torch.flatten(i) for i in grads], dim=0)
norm_factor = grads_flatten.max()
for idx in range(len(grads)):
  grads[idx] /= norm_factor

# for idx, grad in enumerate(grads):
#   print(idx, grad.size())

accu_mode, norm = 'max', 'max'

prune_mask = channel_prune(grads, channel_sparsity=channel_sparsity,
                           accu_mode=accu_mode, norm=norm)
prune_mask_clean = remove_redundant(prune_mask, prune_mode='channel')
do_statistics(prune_mask, prune_mask_clean)

mp_mask = message_passing_prune(grads, channel_sparsity=channel_sparsity,
                                penalty=10, accu_mode=accu_mode, norm=norm)
do_statistics(prune_mask, mp_mask)
print('=> MP and channel prune clean cmp: {}'.format(check_same(mp_mask, prune_mask_clean)))
for jj in range(len(mp_mask)):
  if not torch.equal(mp_mask[jj], prune_mask_clean[jj]):
    print('AW', len(mp_mask), mp_mask[jj].size(), jj, prune_mask_clean[jj].flatten().sum(), mp_mask[jj].flatten().sum())
