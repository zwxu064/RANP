import torch.nn as nn
import copy
import numpy as np


def refine_model(model, neuron_mask_clean, enable_3dunet=False, width=2, network_name='3dunet'):
  refined_model = copy.deepcopy(model)
  layer_idx = 0
  former_conv_valid_neurons = 0

  n_layers = len(neuron_mask_clean) // 2
  # 1:last layer, 2:first two layers, 4:2*2, double_conv, encoders+decoders
  number_of_encoders = (n_layers - 1 - width) // (2 * width)
  last_layer_last_encoder = number_of_encoders * width + width - 1

  for key, layer in refined_model.named_modules():
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv3d,
                          nn.ConvTranspose2d, nn.ConvTranspose3d)):
      weight_mask = neuron_mask_clean[2 * layer_idx]
      bias_mask = neuron_mask_clean[2 * layer_idx + 1]

      assert weight_mask.size() == layer.weight.size()
      assert bias_mask.size() == layer.bias.size() if (bias_mask is not None) else True

      neuron_no, c_in, h, w, d = weight_mask.size()
      valid_neuron_no = (weight_mask.view(neuron_no, -1).sum(1) > 0).sum().cpu().numpy().item()

      # Redefine weight and bias if neurons are pruned
      if (valid_neuron_no != neuron_no) or (former_conv_valid_neurons != c_in):
        if enable_3dunet or (network_name == '3dunet'):
          if True:
            idx_of_concat = ((layer_idx - 1) - last_layer_last_encoder) / width
            if idx_of_concat.is_integer() and (idx_of_concat >= 0) and ((layer_idx - 1) < n_layers - 1 - width):
              concat_layer_idx = (layer_idx - 1) - width - idx_of_concat * 2 * width
              concat_layer_idx = int(concat_layer_idx)
              weight_mask_concat = neuron_mask_clean[concat_layer_idx * 2]
            else:
              weight_mask_concat = None
          else:
            if layer_idx == 8:
              weight_mask_concat = neuron_mask_clean[5 * 2]
            elif layer_idx == 10:
              weight_mask_concat = neuron_mask_clean[3 * 2]
            elif layer_idx == 12:
              weight_mask_concat = neuron_mask_clean[1 * 2]
            else:
              weight_mask_concat = None

          if weight_mask_concat is not None:
            valid_neuron_no_concat = (weight_mask_concat.view(weight_mask_concat.size(0), -1).sum(1) > 0).sum().cpu().numpy().item()
            in_c_refined = former_conv_valid_neurons + valid_neuron_no_concat
          else:
            in_c_refined = c_in if (former_conv_valid_neurons == 0) else former_conv_valid_neurons
        else:
          in_c_refined = c_in if (former_conv_valid_neurons == 0) else former_conv_valid_neurons

        layer.weight = nn.Parameter(layer.weight.new_zeros(valid_neuron_no, in_c_refined, h, w, d), requires_grad=True)
        layer.out_channels = valid_neuron_no
        layer.in_channels = in_c_refined

        if layer.bias is not None:
          layer.bias = nn.Parameter(layer.bias.new_zeros(valid_neuron_no), requires_grad=True)

      former_conv_valid_neurons = valid_neuron_no
      layer_idx += 1

    # Change corresponding batch norm when necessary
    if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
      n_neurons = layer.weight.size(0)
      if (former_conv_valid_neurons > 0) and (n_neurons != former_conv_valid_neurons):
        layer.weight = nn.Parameter(layer.weight.new_zeros(former_conv_valid_neurons), requires_grad=True)
        layer.num_features = former_conv_valid_neurons
        layer.running_mean = layer.running_mean.new_zeros(former_conv_valid_neurons)
        layer.running_var = layer.running_mean.new_ones(former_conv_valid_neurons)

        if layer.bias is not None:
          layer.bias = nn.Parameter(layer.bias.new_zeros(former_conv_valid_neurons), requires_grad=True)

    if isinstance(layer, (nn.GroupNorm)):
      n_neurons = layer.weight.size(0)
      if (former_conv_valid_neurons > 0) and (n_neurons != former_conv_valid_neurons):
        layer.weight = nn.Parameter(layer.weight.new_zeros(former_conv_valid_neurons), requires_grad=True)
        layer.num_channels = former_conv_valid_neurons

        if layer.bias is not None:
          layer.bias = nn.Parameter(layer.bias.new_zeros(former_conv_valid_neurons), requires_grad=True)

  return refined_model


# TODO : network_connection_dict indicates the layer indices for concatenation or addition to simplify the refinement
def refine_model_classification(model, neuron_mask_clean, network_name, network_connection_dict=None,
                                enable_raw_grad=False):
  refined_model = copy.deepcopy(model)
  layer_idx = 0
  former_conv_valid_neurons = 0
  n_layers = len(neuron_mask_clean) // 2
  valid_neuron_list = []

  # Valid neuron number in every layer
  for idx in range(n_layers):
    current_layer = neuron_mask_clean[2 * idx]
    valid_neuron = int((current_layer.view(current_layer.size(0), -1).sum(1) > 0).sum().cpu().numpy().item())
    valid_neuron_list.append(valid_neuron)

  for key, layer in refined_model.named_modules():
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv3d,
                          nn.ConvTranspose2d, nn.ConvTranspose3d)):
      weight_mask = neuron_mask_clean[2 * layer_idx]
      bias_mask = neuron_mask_clean[2 * layer_idx + 1]
      valid_neuron_no = valid_neuron_list[layer_idx]

      assert weight_mask.size() == layer.weight.size()
      assert bias_mask.size() == layer.bias.size() if (bias_mask is not None) else True

      weight_len = len(weight_mask.size())
      if weight_len == 5:
        neuron_no, c_in, h, w, d = weight_mask.size()
      elif weight_len == 2:
        neuron_no, c_in = weight_mask.size()

      # Redefine weight and bias if neurons are pruned
      # Debug : the last one is a patch since conv layer having groups many has valid_neuron_no == neuron_no if all retained
      if (valid_neuron_no != neuron_no) or (former_conv_valid_neurons != c_in) \
              or hasattr(layer, 'groups'):
        in_c_refined = c_in if (layer_idx == 0) else valid_neuron_list[layer_idx - 1]

        if hasattr(layer, 'groups') and (layer.groups > 1):
          # Just a patch for mobilenetv2 MNMG, too low acc. so increase valid_neuron_no to increase
          if enable_raw_grad:
            valid_neuron_no_tmp = int(in_c_refined * (np.ceil(valid_neuron_no / in_c_refined)))
            # In case that FLOPs is larger than the full one by using just ceil
            if valid_neuron_no_tmp > layer.out_channels:
              valid_neuron_no_tmp = int(in_c_refined * (np.floor(valid_neuron_no / in_c_refined)))
            valid_neuron_no = valid_neuron_no_tmp
          else:
            # Debug, note that it makes valid_neuron_no useless but only depends on factor * groups, leading to many 1s and low acc.
            # Note that out_channels = org_factor * groups
            org_factor = layer.out_channels // layer.in_channels  # Zhiwei : When having groups, in_c and o_c must be dividable by groups
            valid_neuron_no = int(org_factor * in_c_refined)

          layer.groups = in_c_refined

          if weight_len == 5:
            layer.weight = nn.Parameter(layer.weight.new_zeros(valid_neuron_no, 1, h, w, d), requires_grad=True)
          elif weight_len == 2:
            layer.weight = nn.Parameter(layer.weight.new_zeros(valid_neuron_no, 1), requires_grad=True)
            layer.in_features = in_c_refined
            layer.out_features = valid_neuron_no
        else:
          # Check if concate or sum with other layers, such as residual type
          # This is sum type:
          # connected_layer_idx_list = searching_connections(layer_idx, network_connection_dict)
          connected_layer_idx_list = [layer_idx]

          def searching_connections(layer_idx, network_connection_dict, connected_layer_idx_list):
            for connected_layer_idx in network_connection_dict[str(layer_idx)]['connection_layer']:
              connected_layer_idx_list.append(connected_layer_idx)
              if connected_layer_idx > layer_idx:
                connected_layer_idx_list = searching_connections(connected_layer_idx,
                                                                 network_connection_dict,
                                                                 connected_layer_idx_list)
            return connected_layer_idx_list

          connected_layer_idx_list = searching_connections(layer_idx, network_connection_dict, connected_layer_idx_list)
          connected_layer_idx_list = np.unique(connected_layer_idx_list)
          valid_neuron_no = int(np.max([valid_neuron_list[v] for v in connected_layer_idx_list]))

          if weight_len == 5:
            layer.weight = nn.Parameter(layer.weight.new_zeros(valid_neuron_no, in_c_refined, h, w, d), requires_grad=True)
          elif weight_len == 2:
            layer.weight = nn.Parameter(layer.weight.new_zeros(valid_neuron_no, in_c_refined), requires_grad=True)
            layer.in_features = in_c_refined
            layer.out_features = valid_neuron_no

        valid_neuron_list[layer_idx] = valid_neuron_no
        layer.in_channels = in_c_refined
        layer.out_channels = valid_neuron_no

        if layer.bias is not None:
          layer.bias = nn.Parameter(layer.bias.new_zeros(valid_neuron_no), requires_grad=True)

      former_conv_valid_neurons = valid_neuron_no
      layer_idx += 1

    # Change corresponding batch norm when necessary
    if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
      n_neurons = layer.weight.size(0)
      if (former_conv_valid_neurons > 0) and (n_neurons != former_conv_valid_neurons):
        layer.weight = nn.Parameter(layer.weight.new_zeros(former_conv_valid_neurons), requires_grad=True)
        layer.num_features = former_conv_valid_neurons
        layer.running_mean = layer.running_mean.new_zeros(former_conv_valid_neurons)
        layer.running_var = layer.running_mean.new_ones(former_conv_valid_neurons)

        if layer.bias is not None:
          layer.bias = nn.Parameter(layer.bias.new_zeros(former_conv_valid_neurons), requires_grad=True)

    if isinstance(layer, (nn.GroupNorm)):
      n_neurons = layer.weight.size(0)
      if (former_conv_valid_neurons > 0) and (n_neurons != former_conv_valid_neurons):
        layer.weight = nn.Parameter(layer.weight.new_zeros(former_conv_valid_neurons), requires_grad=True)
        layer.num_channels = former_conv_valid_neurons

        if layer.bias is not None:
          layer.bias = nn.Parameter(layer.bias.new_zeros(former_conv_valid_neurons), requires_grad=True)

  return refined_model


def refine_model_I3D(model, neuron_mask_clean):
  refined_model = copy.deepcopy(model)
  n_layers = len(neuron_mask_clean) // 2
  valid_neuron_list = []

  for idx in range(n_layers):
    current_layer = neuron_mask_clean[2 * idx]
    kernel_size = current_layer.size(2)
    out_c = (current_layer.view(current_layer.size(0), -1).sum(1) != 0).float().sum()
    in_c = (current_layer.transpose(0, 1).contiguous().view(current_layer.size(1), -1).sum(1) != 0).float().sum()
    valid_neuron_list.append([int(out_c.cpu().numpy().item()),
                              int(in_c.cpu().numpy().item()),
                              int(kernel_size)])

  # Check
  layer_idx = 0
  for key, layer in refined_model.named_modules():
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv3d,
                          nn.ConvTranspose2d, nn.ConvTranspose3d)):
      weight_mask = neuron_mask_clean[2 * layer_idx]
      bias_mask = neuron_mask_clean[2 * layer_idx + 1]

      assert weight_mask.size() == layer.weight.size()
      assert bias_mask.size() == layer.bias.size() if (bias_mask is not None) else True
      layer_idx += 1

  layer_idx = 0
  for key, layer in refined_model.named_modules():
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv3d,
                          nn.ConvTranspose2d, nn.ConvTranspose3d)):
      weight_len = len(layer.weight.size())
      out_c, in_c, ksz = valid_neuron_list[layer_idx]

      if weight_len == 5:
        layer.weight = nn.Parameter(
          layer.weight.new_zeros(out_c, in_c, ksz, ksz, ksz),
          requires_grad=True)
      elif weight_len == 2:
        layer.weight = nn.Parameter(
          layer.weight.new_zeros(out_c, in_c),
          requires_grad=True)
        layer.in_features = in_c
        layer.out_features = out_c

      layer.in_channels = in_c
      layer.out_channels = out_c

      if layer.bias is not None:
        layer.bias = nn.Parameter(layer.bias.new_zeros(out_c),
                                  requires_grad=True)

      layer_idx += 1

    if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
      out_c = valid_neuron_list[layer_idx - 1][0]
      layer.weight = nn.Parameter(layer.weight.new_zeros(out_c),
                                  requires_grad=True)
      layer.num_features = out_c
      layer.running_mean = layer.running_mean.new_zeros(out_c)
      layer.running_var = layer.running_mean.new_ones(out_c)

      if layer.bias is not None:
        layer.bias = nn.Parameter(layer.bias.new_zeros(out_c),
                                  requires_grad=True)

  return refined_model


def refine_model_PSM(model, valid_neuron_list):
  refined_model = copy.deepcopy(model)

  # ===========================================
  layer_idx = 0
  for key, layer in refined_model.named_modules():
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv3d,
                          nn.ConvTranspose2d, nn.ConvTranspose3d)):
      if isinstance(layer, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
        in_c, out_c, ksz, klen = valid_neuron_list[layer_idx]  # This should be paied attention
      else:
        out_c, in_c, ksz, klen = valid_neuron_list[layer_idx]  # This is normal

      if klen == 5:
        layer.weight = nn.Parameter(
          layer.weight.new_zeros(out_c, in_c, ksz, ksz, ksz),
          requires_grad=True)
      elif klen == 4:
          layer.weight = nn.Parameter(
            layer.weight.new_zeros(out_c, in_c, ksz, ksz),
            requires_grad=True)
      elif klen == 2:
        layer.weight = nn.Parameter(
          layer.weight.new_zeros(out_c, in_c),
          requires_grad=True)
        layer.in_features = in_c
        layer.out_features = out_c

      layer.in_channels = in_c
      layer.out_channels = out_c

      if layer.bias is not None:
        layer.bias = nn.Parameter(layer.bias.new_zeros(out_c),
                                  requires_grad=True)

      layer_idx += 1

    if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
      out_c = valid_neuron_list[layer_idx - 1][0]
      layer.weight = nn.Parameter(layer.weight.new_zeros(out_c),
                                  requires_grad=True)
      layer.num_features = out_c
      layer.running_mean = layer.running_mean.new_zeros(out_c)
      layer.running_var = layer.running_mean.new_ones(out_c)

      if layer.bias is not None:
        layer.bias = nn.Parameter(layer.bias.new_zeros(out_c),
                                  requires_grad=True)

  return refined_model