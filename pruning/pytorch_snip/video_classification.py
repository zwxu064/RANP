import torch
import copy
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import json
import os


def remove_redundant_3dmobilenet(keep_masks):
  if keep_masks is None:
    return

  keep_masks = copy.deepcopy(keep_masks)
  # print('=' * 20, 'remove redundant retains', '=' * 20)
  n_layers = len(keep_masks) // 2

  # Forward
  for l in range(n_layers - 1):
    weight_idx, bias_idx = 2 * l, 2 * l + 1
    next_weight_idx, next_bias_idx = 2 * (l + 1), 2 * (l + 1) + 1

    current_layer_mask = keep_masks[weight_idx]
    next_layer_mask = keep_masks[next_weight_idx]
    next_layer_bias_mask = keep_masks[next_bias_idx]
    current_layer_out_c = current_layer_mask.size(0)
    following_in_c = next_layer_mask.size(1)

    if following_in_c == 1:  # TODO : it is better to use groups>1, special case for conv group > 1
      continue
    else:
      assert current_layer_out_c == following_in_c

    for idx_neuron in range(current_layer_out_c):
      # All conv3d except the last one have no bias
      if (current_layer_mask[idx_neuron, :].sum() == 0) \
              and (next_layer_mask[:, idx_neuron].sum() != 0):
        invalid_area = next_layer_mask[:, idx_neuron] == 1
        next_layer_mask[:, idx_neuron][invalid_area] = 0

        if next_layer_bias_mask is not None:
          invalid_area = next_layer_mask.view(next_layer_mask.size(0), -1).sum(1) == 0
          next_layer_bias_mask[invalid_area] = 0

  # Backward
  for l in range(n_layers - 1, 0, -1):
    weight_idx, bias_idx = 2 * l, 2 * l + 1
    front_weight_idx, front_bias_idx = 2 * (l - 1), 2 * (l - 1) + 1
    current_layer_mask = keep_masks[weight_idx]
    front_layer_mask = keep_masks[front_weight_idx]
    front_layer_bias_mask = keep_masks[front_bias_idx]
    front_layer_out_c = front_layer_mask.size(0)
    current_layer_in_c = current_layer_mask.size(1)

    if current_layer_in_c == 1:  # TODO : same above that use groups>1
      continue
    else:
      assert current_layer_in_c == front_layer_out_c

    for idx_neuron in range(current_layer_in_c):
      if (current_layer_mask[:, idx_neuron].sum() == 0) \
              and (front_layer_mask[idx_neuron, :].sum() != 0):
        invalid_area = front_layer_mask[idx_neuron, :] == 1
        front_layer_mask[idx_neuron, :][invalid_area] = 0

        if front_layer_bias_mask is not None:
          front_layer_bias_mask[idx_neuron] = 0

  return keep_masks


def create_network_connection_dict(model, json_network_connection=None):
  json_path = json_network_connection
  if json_path is not None:
    json_path_modified = json_network_connection.replace('.json', '_modified.json')

    if os.path.exists(json_path_modified):
      with open(json_path_modified, 'r') as f:
        network_connection_dict = json.load(f)
    else:  # Build an empty form
      network_connection_dict = {}

      layer_count = 0
      for idx, (key, layer) in enumerate(model.named_modules()):
        if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv3d,
                              nn.ConvTranspose2d, nn.ConvTranspose3d)):
          network_connection_dict[str(layer_count)] = {
            'key': key,
            'shape':list(layer.weight.shape),
            'groups': layer.groups if hasattr(layer, 'groups') else [],
            'connection_layer': [],
            'bias': 1 if (layer.bias is not None) else 0,
          }
          layer_count += 1

      with open(json_path, 'w') as f:
        json.dump(network_connection_dict, f)
  else:
    network_connection_dict = None

  return network_connection_dict


def remove_redundant_I3D(keep_masks, encoder_layer_index=None):
  # Neuron pruning just on forward passing to set in_channels
  # For channel and weight pruning, need to do forward and backward passing
  if keep_masks is None:
    return

  keep_masks = copy.deepcopy(keep_masks)
  n_layers = len(keep_masks) // 2

  if encoder_layer_index is None:
    assert n_layers == 58
    encoder_layer_index = [3, 9, 15, 21, 27, 33, 39, 45, 51]

  skip_list = [v for v in encoder_layer_index]
  skip_list += [v + 2 for v in encoder_layer_index]
  skip_list += [v + 4 for v in encoder_layer_index]
  skip_list += [v + 5 for v in encoder_layer_index]
  former_merge_out_c_list = []

  # Forward
  for l in range(n_layers - 1):
    weight_idx, bias_idx = 2 * l, 2 * l + 1
    next_weight_idx, next_bias_idx = 2 * (l + 1), 2 * (l + 1) + 1

    current_layer_mask = keep_masks[weight_idx]
    next_layer_mask = keep_masks[next_weight_idx]
    current_layer_out_c = current_layer_mask.size(0)
    next_layer_in_c = next_layer_mask.size(1)

    if (l + 1) not in encoder_layer_index:  # regular
      if (l in skip_list) and (l != 56):
        continue
      elif l == 56:
        for neuron_idx in range(len(former_merge_out_c_list)):
          if former_merge_out_c_list[neuron_idx] == 0:
            keep_masks[2 * (l + 1)][:, neuron_idx] = 0  # last layer

        continue

      assert current_layer_out_c == next_layer_in_c

      if len(next_layer_mask.shape) == 4:
        current_layer_mask_trans = current_layer_mask.view(current_layer_out_c, -1, 1, 1)
      elif len(next_layer_mask.shape) == 5:
        current_layer_mask_trans = current_layer_mask.view(current_layer_out_c, -1, 1, 1, 1)
      else:
        current_layer_mask_trans = current_layer_mask.view(current_layer_out_c, -1)

      current_layer_mask_trans = (current_layer_mask_trans.sum(1, keepdim=True) != 0).transpose(0, 1)

      # Only for nonzero neurons, since some neurons are previously marked 0 in pruning but not here
      # This also makes sparse mask dense for param pruning by SNIP
      next_layer_mask_nonzero = (next_layer_mask.view(next_layer_mask.shape[0], -1).sum(1) != 0)
      next_layer_mask[next_layer_mask_nonzero] = current_layer_mask_trans.float()  # make it dense

      # This is used in non-SNIP in the paper, but it should be the same as above
      # This is okay for neuron pruning
      # for neuron_idx in range(current_layer_out_c):
      #   if current_layer_mask[neuron_idx].sum() == 0:  # forward passing
      #     next_layer_mask[:, neuron_idx] = 0

      # addtion: backward passing, good for one layer but I3D has 6 layer as a module, too complex, do not waste time
      # if next_layer_mask[:, neuron_idx].sum() == 0:
      #   current_layer_mask[neuron_idx] = 0
    else:  # in a 6-layer encoder
      if l == 2:
        former_merge_out_c_list = (current_layer_mask.view(current_layer_out_c, -1).sum(1) != 0)
        former_merge_out_c_list = former_merge_out_c_list.float()

      next_layer_0 = keep_masks[2 * (l + 1)]
      next_layer_1 = keep_masks[2 * (l + 2)]
      next_layer_2 = keep_masks[2 * (l + 3)]
      next_layer_3 = keep_masks[2 * (l + 4)]
      next_layer_4 = keep_masks[2 * (l + 5)]
      next_layer_5 = keep_masks[2 * (l + 6)]

      assign_list, merge_list = [], []
      assign_list.append(next_layer_0)
      assign_list.append(next_layer_1)
      assign_list.append(next_layer_3)
      assign_list.append(next_layer_5)

      merge_list.append(next_layer_0)
      merge_list.append(next_layer_2)
      merge_list.append(next_layer_4)
      merge_list.append(next_layer_5)

      for ii in assign_list:
        tmp = former_merge_out_c_list.shape[0]

        if len(ii.shape) == 4:
          former_merge_trans = former_merge_out_c_list.view(tmp, -1, 1, 1)
        elif len(ii.shape) == 5:
          former_merge_trans = former_merge_out_c_list.view(tmp, -1, 1, 1, 1)
        else:
          former_merge_trans = former_merge_out_c_list.view(tmp, -1)

        former_merge_trans = (former_merge_trans.sum(1, keepdim=True) != 0).transpose(0, 1)

        # Only for nonzero neurons, since some neurons are previously marked 0 in pruning but not here
        # This also makes sparse mask dense for param pruning by SNIP
        ii_nonzero = (ii.view(ii.shape[0], -1).sum(1) != 0)
        ii[ii_nonzero] = former_merge_trans.float()  # make it dense

      # This is used in non-SNIP in the paper, but it should be the same as above
      # for neuron_idx in range(len(former_merge_out_c_list)):
      #   if former_merge_out_c_list[neuron_idx] == 0:
      #     next_layer_0[:, neuron_idx] = 0
      #     next_layer_1[:, neuron_idx] = 0
      #     next_layer_3[:, neuron_idx] = 0
      #     next_layer_5[:, neuron_idx] = 0

      former_merge_out_c_list = []
      for ii in merge_list:
        former_merge_out_c_list.append(ii.view(ii.size(0), -1).sum(1) != 0)

      former_merge_out_c_list = torch.cat(former_merge_out_c_list, dim=0)
      former_merge_out_c_list = former_merge_out_c_list.float()

  return keep_masks