import torch, os, time, copy, types, sys
import numpy as np
import scipy.io as scio
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('./pruning/pytorch_snip')
from prune_utils import check_same, convert_dim_conv2fully, resume_dim_conv2fully, cal_channel_prune_grad
from mp_prune import message_passing_prune
from video_classification import remove_redundant_3dmobilenet, remove_redundant_I3D
from video_classification import remove_redundant_PSM
from torch.autograd import Variable


enable_verbose = False

def do_statistics_model(model):
  n_params, n_neurons = 0, 0
  for key, layer in model.named_modules():
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv3d,
                          nn.ConvTranspose2d, nn.ConvTranspose3d)):
      n_params += layer.weight.data.flatten().size(0)
      n_neurons += layer.weight.data.size(0)

      if layer.bias is not None:
        n_params += layer.bias.data.size(0)

  return n_params, n_neurons


def dump_neuron_per_layer(full_model, refined_model):
  full_neuron_layer, refined_neuron_layer = [], []

  for key, layer in full_model.named_modules():
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv3d,
                          nn.ConvTranspose2d, nn.ConvTranspose3d)):
      full_neuron_layer.append(layer.weight.data.size(0))

  for key, layer in refined_model.named_modules():
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv3d,
                          nn.ConvTranspose2d, nn.ConvTranspose3d)):
      refined_neuron_layer.append(layer.weight.data.size(0))

  scio.savemat('dump/neuron_per_layer_list.mat',
               {'full_model': full_neuron_layer,
                'refined_model': refined_neuron_layer})


def update_grads_average(grads_abs_average, grads_abs, batch_idx):
  if len(grads_abs_average) == 0:
    assert batch_idx == 0
    grads_abs_average = grads_abs
  else:
    num = len(grads_abs_average)
    assert len(grads_abs) == num
    assert batch_idx >= 1
    for idx in range(num):
      if grads_abs[idx] is not None:
        grads_abs_average[idx] = (grads_abs_average[idx] * batch_idx + grads_abs[idx]) / (batch_idx + 1)

  return grads_abs_average


def snip_forward_linear(self, x):
  bias = self.bias
  weight = self.weight

  if hasattr(self, 'bias_mask'):
    bias = self.bias * self.bias_mask

  if hasattr(self, 'weight_mask'):
    weight = self.weight * self.weight_mask

  output = F.linear(x, weight, bias)

  if hasattr(self, 'hidden_mask'):
    output = output * self.hidden_mask

  return output


def snip_forward_conv2d(self, x):
  bias = self.bias
  weight = self.weight

  if hasattr(self, 'bias_mask'):
    bias = self.bias * self.bias_mask

  if hasattr(self, 'weight_mask'):
    weight = self.weight * self.weight_mask

  output = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

  if hasattr(self, 'hidden_mask'):
    output = output * self.hidden_mask

  return output


def snip_forward_conv3d(self, x):
  bias = self.bias
  weight = self.weight

  if hasattr(self, 'bias_mask'):
    bias = self.bias * self.bias_mask

  if hasattr(self, 'weight_mask'):
    weight = self.weight * self.weight_mask

  output = F.conv3d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

  if hasattr(self, 'hidden_mask'):
    output = output * self.hidden_mask

  return output


def snip_forward_conv3dtranspose(self, x):
  bias = self.bias
  weight = self.weight

  if hasattr(self, 'bias_mask'):
    bias = self.bias * self.bias_mask

  if hasattr(self, 'weight_mask'):
    weight = self.weight * self.weight_mask

  output = F.conv_transpose3d(x, weight, bias,
                              self.stride, self.padding,
                              output_padding=self.output_padding,
                              dilation=self.dilation,
                              groups=self.groups)

  if hasattr(self, 'hidden_mask'):
    output = output * self.hidden_mask

  return output


def add_mask_for_hidden_hook(self, input, output):
  batch, c, h, w, d = output.size()
  self.hidden_mask = nn.Parameter(torch.ones((c, h, w, d),
                                             dtype=output.dtype,
                                             device=output.device,
                                             requires_grad=True))


def add_mask_for_grad(net, args, enable_kernel_mask=True, enable_hidden_mask=False):
  # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
  # instead of the weights
  # Zhiwei instead of using random one batch, using the whole dataset to get average grads of mask
  net = net.cpu()

  for layer in net.modules():
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
      # This is for reproducing, will affect pruning as well as training
      # torch.manual_seed(0)
      # torch.cuda.manual_seed(0)
      # torch.cuda.manual_seed_all(0)

      if enable_kernel_mask:
        layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))

      # New function Zhiwei
      if enable_hidden_mask:
        layer.hidden_mask_hook = layer.register_forward_hook(add_mask_for_hidden_hook)

      if args.weight_init == 'xn':
        nn.init.xavier_normal_(layer.weight)
      elif args.weight_init == 'xu':
        nn.init.xavier_uniform_(layer.weight)
      elif args.weight_init == 'kn':
        nn.init.kaiming_normal_(layer.weight)
      elif args.weight_init == 'ku':
        nn.init.kaiming_uniform_(layer.weight)
      elif args.weight_init in ['orthogonal', 'ort']:
        nn.init.orthogonal_(layer.weight)
      elif args.weight_init in ['one', 'fixed']:
        nn.init.constant_(layer.weight, 1)
      else:
        assert False

      layer.weight.requires_grad = False  # Cuz it is fixed by initialization

      if layer.bias is not None:
        if enable_kernel_mask:
          layer.bias_mask = nn.Parameter(torch.ones_like(layer.bias))
        nn.init.zeros_(layer.bias)
        layer.bias.requires_grad = False

    # Bug this is important for reproducing
    if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
      if layer.weight is not None:
        # not good, this will make channel prune remove whole layers
        # nn.init.constant_(layer.weight, 1)
        nn.init.uniform_(layer.weight)

      if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

    # Override the forward methods:
    if isinstance(layer, nn.Linear):
      layer.forward = types.MethodType(snip_forward_linear, layer)

    if isinstance(layer, nn.Conv2d):
      layer.forward = types.MethodType(snip_forward_conv2d, layer)

    if isinstance(layer, nn.Conv3d):
      layer.forward = types.MethodType(snip_forward_conv3d, layer)

    if isinstance(layer, nn.ConvTranspose3d):
      layer.forward = types.MethodType(snip_forward_conv3dtranspose, layer)

  return net.cuda()


def get_mask_grad(net, enable_kernel_mask=True, enable_hidden_mask=False, enable_raw_grad=False):
  kernel_mask_grads_abs, hidden_mask_grads_abs = [], []

  for layer in net.modules():
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
      if enable_kernel_mask:
        if enable_raw_grad:
          kernel_mask_grads_abs.append(layer.weight_mask.grad)
        else:
          kernel_mask_grads_abs.append(torch.abs(layer.weight_mask.grad))

        if layer.bias is not None:
          if enable_raw_grad:
            kernel_mask_grads_abs.append(layer.bias_mask.grad)
          else:
            kernel_mask_grads_abs.append(torch.abs(layer.bias_mask.grad))
        else:
          kernel_mask_grads_abs.append(None)

      if enable_hidden_mask:
        if enable_raw_grad:
          hidden_mask_grads_abs.append(layer.hidden_mask.grad)
        else:
          hidden_mask_grads_abs.append(torch.abs(layer.hidden_mask.grad))

  return kernel_mask_grads_abs, hidden_mask_grads_abs


# ==== For hidden layer pruning
def apply_forward_hidden_mask_hook(self, input, output):
  # output.data[self.hidden_mask.unsqueeze(0) == 0.] = 0
  if output.data.device != self.hidden_mask.device:
    self.hidden_mask = self.hidden_mask.to(output.data.device)

  output.data = output.data * self.hidden_mask


def apply_backward_hidden_mask_hook(self, grad_in, grad_out):
  grad_out[0].data = grad_out[0].data * self.hidden_mask


def remove_hooks(net):
  for layer in net.modules():
    if hasattr(layer, 'hidden_mask'):
      layer.hidden_mask_hook.remove()


def apply_hidden_mask(net, hidden_masks, enable_hidden_sum=False):
  prunable_layers = filter(lambda layer: isinstance(layer, (nn.Linear,
                                                            nn.Conv2d,
                                                            nn.Conv3d,
                                                            nn.ConvTranspose3d)),
                           net.modules())

  for idx, (layer, hidden_mask) in enumerate(zip(prunable_layers, hidden_masks)):
    if not hasattr(layer, 'hidden_mask'):
      layer.hidden_mask = hidden_mask

      if enable_hidden_sum:
        layer.hidden_mask = layer.hidden_mask.unsqueeze(0)

    if not hasattr(layer, 'hidden_mask_forward_hook'):
      layer.hidden_mask_forward_hook = layer.register_forward_hook(apply_forward_hidden_mask_hook)

    if not hasattr(layer, 'hidden_mask_backward_hook'):
      layer.hidden_mask_backward_hook = layer.register_backward_hook(apply_backward_hidden_mask_hook)


def apply_prune_mask(net, keep_masks):
  prunable_layers = filter(lambda layer: isinstance(layer, (nn.Conv2d,
                                                            nn.Linear,
                                                            nn.Conv3d,
                                                            nn.ConvTranspose3d)),
                           net.modules())

  for idx, layer in enumerate(prunable_layers):
    weight_mask, bias_mask = keep_masks[2 * idx], keep_masks[2 * idx + 1]
    assert (layer.weight.shape == weight_mask.shape)

    def hook_factory(mask):
      """
      The hook function can't be defined directly here because of Python's
      late binding which would result in all hooks getting the very last
      mask! Getting it through another function forces early binding.
      """

      def hook(grads):
          return grads * mask

      return hook

    # mask[i] == 0 --> Prune parameter
    # mask[i] == 1 --> Keep parameter
    # Step 1: Set the masked weights/bias to zero
    # Step 2: Make sure their gradients remain zero
    layer.weight.data[weight_mask == 0.] = 0.
    layer.weight.register_hook(hook_factory(weight_mask))

    if bias_mask is not None:
      assert (layer.bias.shape == bias_mask.shape)
      layer.bias.data[bias_mask == 0.] = 0.
      layer.bias.register_hook(hook_factory(bias_mask))


# ==== 3DUNet pruning
def apply_prune_mask_3dunet(net, keep_masks):
  prunable_layers = filter(lambda layer: isinstance(layer, (nn.Conv2d,
                                                            nn.Linear,
                                                            nn.Conv3d,
                                                            nn.ConvTranspose3d)),
                           net.modules())

  for idx, (layer) in enumerate(prunable_layers):
    weight_mask, bias_mask = keep_masks[2 * idx], keep_masks[2 * idx + 1]
    assert (layer.weight.shape == weight_mask.shape)

    def hook_factory(mask):
      """
      The hook function can't be defined directly here because of Python's
      late binding which would result in all hooks getting the very last
      mask! Getting it through another function forces early binding.
      """

      def hook(grads):
          return grads * mask

      return hook

    # mask[i] == 0 --> Prune parameter
    # mask[i] == 1 --> Keep parameter
    # Step 1: Set the masked weights/bias to zero
    # Step 2: Make sure their gradients remain zero
    layer.weight.data[weight_mask == 0.] = 0.
    layer.weight.register_hook(hook_factory(weight_mask))

    if bias_mask is not None:
      assert (layer.bias.shape == bias_mask.shape)
      layer.bias.data[bias_mask == 0.] = 0.
      layer.bias.register_hook(hook_factory(bias_mask))


def do_statistics(retain_mask, clean_mask):
  assert len(retain_mask) == len(clean_mask)
  n_layers = len(retain_mask) // 2
  valid_per_layer, invalid_per_layer, retain_per_layer, all_per_layer = [], [], [], []
  n_invalids, n_retains, n_all = 0, 0, 0

  for idx in range(n_layers):
    weight_idx, bias_idx = 2 * idx, 2 * idx + 1
    retain_layer, clean_layer = retain_mask[weight_idx], clean_mask[weight_idx]
    assert retain_layer.size() == clean_layer.size()
    retain_sum = retain_layer.sum()
    valid_sum = clean_layer.sum()
    all_sum = (retain_layer >= 0).float().sum()

    if retain_mask[bias_idx] is not None:
      retain_sum += retain_mask[bias_idx].sum()
      valid_sum += clean_mask[bias_idx].sum()
      all_sum += (retain_mask[bias_idx] >= 0).float().sum()

    invalid_sum = retain_sum - valid_sum
    n_invalids += invalid_sum
    n_retains += retain_sum
    n_all += all_sum

    retain_per_layer.append(torch.tensor([retain_sum]))
    valid_per_layer.append(torch.tensor([valid_sum]))
    invalid_per_layer.append(torch.tensor([invalid_sum]))
    all_per_layer.append(torch.tensor([all_sum]))

  retain_per_layer = torch.cat(retain_per_layer, dim=0)
  valid_per_layer = torch.cat(valid_per_layer, dim=0)
  invalid_per_layer = torch.cat(invalid_per_layer, dim=0)
  all_per_layer = torch.cat(all_per_layer, dim=0)
  invalid_by_retained = invalid_per_layer / retain_per_layer
  valid_by_total = valid_per_layer / all_per_layer

  if False:  # enable manuually, otherwise a mess
    print('valid in layer: {}'.format(valid_per_layer.int().cpu().numpy()))
    print('invalid in layer: {}'.format(invalid_per_layer.int().cpu().numpy()))
    print('retain in layer: {}'.format(retain_per_layer.int().cpu().numpy()))
    print('total in layer: {}'.format(all_per_layer.int().cpu().numpy()))
    print('invalid/retain in layer: {}'.format(invalid_by_retained.cpu().numpy()))
    print('valid/total in layer: {}'.format(valid_by_total.cpu().numpy()))

    if enable_verbose:
      if (n_retains > 0) and (n_all > 0):
        print('invalid: {}, retain: {}, all: {}\ninvalid/retain: {:.4f}, retain/all: {:.4f}, valid/all: {:.4f}' \
              .format(n_invalids.int(), n_retains.int(), n_all.int(),
                      float(n_invalids) / float(n_retains),
                      float(n_retains) / float(n_all),
                      float(n_retains - n_invalids) / float(n_all)))

  zero_fill = retain_mask[0].new_zeros(1, dtype=torch.uint8)
  n_params = torch.sum(torch.cat([torch.flatten(mask.view(mask.size(0), -1) >= 0) if (mask is not None) else zero_fill for mask in retain_mask]))
  n_neuron_total = torch.sum(torch.cat([torch.flatten(mask.view(mask.size(0), -1).sum(1) >= 0) if (idx % 2 == 0) else zero_fill for idx, mask in enumerate(retain_mask)]))
  n_neuron_retained = torch.sum(torch.cat([torch.flatten(mask.view(mask.size(0), -1).sum(1) > 0) if (idx % 2 == 0) else zero_fill for idx, mask in enumerate(retain_mask)]))
  n_neuron_clean = torch.sum(torch.cat([torch.flatten(mask.view(mask.size(0), -1).sum(1) > 0) if (idx % 2 == 0) else zero_fill for idx, mask in enumerate(clean_mask)]))

  if enable_verbose:
    print('Num params: {:d}; neuron total: {:d}, after pruning: {:d} ({:.4f}%), after cleaning: {:d} ({:.4f}%)'. \
          format(n_params, n_neuron_total,
                 n_neuron_retained, n_neuron_retained.float() * 100 / n_neuron_total.float(),
                 n_neuron_clean, n_neuron_clean.float() * 100 / n_neuron_total.float()))

  return invalid_per_layer, retain_per_layer, all_per_layer


def remove_redundant_3dunet(keep_masks, width=2):  # only support neuron pruning
  if keep_masks is None:
    return

  keep_masks = copy.deepcopy(keep_masks)

  if enable_verbose:
    print('=' * 20, 'remove redundant retains', '=' * 20)

  n_layers = len(keep_masks) // 2
  # assert n_layers == 15

  # 1:last layer, 2:first two layers, 4:2*2, double_conv, encoders+decoders
  number_of_encoders = (n_layers - 1 - width) // (2 * width)
  last_layer_last_encoder = number_of_encoders * width + width - 1

  # Forward
  for l in range(n_layers - 1):
    weight_idx, bias_idx = 2 * l, 2 * l + 1
    next_weight_idx, next_bias_idx = 2 * (l + 1), 2 * (l + 1) + 1

    current_layer_mask = keep_masks[weight_idx]
    next_layer_mask = keep_masks[next_weight_idx]
    next_layer_bias_mask = keep_masks[next_bias_idx]

    # for the case of concatenation, channels unmatched
    current_layer_out_c = current_layer_mask.size(0)
    following_in_c = next_layer_mask.size(1)
    concatenate_layer_mask = []
    concatenate_layer_out_c = 0

    # Deal with concatenation which causes difference between out and in channels
    if current_layer_out_c != following_in_c:

      if enable_verbose:
        print('Warning (this is fine, concatenation), current layer: {}, following: {}' \
              .format(current_layer_mask.size(), next_layer_mask.size()))

      if True:
        idx_of_concat = (l - last_layer_last_encoder) / width
        if idx_of_concat.is_integer() and (idx_of_concat >= 0) and (l < n_layers - 1 - width):
          concat_layer_idx = l - width - idx_of_concat * 2 * width
          concat_layer_idx = int(concat_layer_idx)
          concatenate_layer_mask = keep_masks[concat_layer_idx * 2]
          concatenate_layer_out_c = concatenate_layer_mask.size(0)
          assert (concatenate_layer_out_c + current_layer_out_c == following_in_c)
      else:
        if l == 7:
          concatenate_layer_mask = keep_masks[5 * 2]
        elif l == 9:
          concatenate_layer_mask = keep_masks[3 * 2]
        elif l == 11:
          concatenate_layer_mask = keep_masks[1 * 2]

        concatenate_layer_out_c = concatenate_layer_mask.size(0)
        assert (concatenate_layer_out_c + current_layer_out_c == following_in_c)

    # Concatenation first, then current_layer
    for idx_neuron in range(concatenate_layer_out_c):
      # All conv3d except the last one have no bias
      if (concatenate_layer_mask[idx_neuron, :, :, :, :].sum() == 0) \
              and (next_layer_mask[:, idx_neuron, :, :, :].sum() != 0):
        invalid_area = next_layer_mask[:, idx_neuron, :, :, :] == 1
        next_layer_mask[:, idx_neuron, :, :, :][invalid_area] = 0

    for idx_neuron in range(current_layer_out_c):
      idx_neuron_concat = idx_neuron + concatenate_layer_out_c

      if (current_layer_mask[idx_neuron, :, :, :, :].sum() == 0) \
              and (next_layer_mask[:, idx_neuron_concat, :, :, :].sum() != 0):
        invalid_area = next_layer_mask[:, idx_neuron_concat, :, :, :] == 1
        next_layer_mask[:, idx_neuron_concat, :, :, :][invalid_area] = 0

        # when weights of a neuron are all removed, its bias should also be removed
        if next_layer_bias_mask is not None:
          invalid_area = next_layer_mask.view(next_layer_mask.size(0), -1).sum(1) == 0
          next_layer_bias_mask[invalid_area] = 0

  # Backward
  for l in range(n_layers - 1, 0, -1):
    # continue
    weight_idx, bias_idx = 2 * l, 2 * l + 1
    front_weight_idx, front_bias_idx = 2 * (l - 1), 2 * (l - 1) + 1
    current_layer_mask = keep_masks[weight_idx]
    front_layer_mask = keep_masks[front_weight_idx]

    # Deal with concatenation which causes difference between out and in channels
    front_layer_out_c = front_layer_mask.size()[0]
    current_layer_in_c = current_layer_mask.size()[1]
    concatenate_layer_out_c = 0

    if current_layer_in_c != front_layer_out_c:
      if enable_verbose:
        print('Warning (this is fine, concatenation), front layer: {}, current layer: {}' \
              .format(front_layer_mask.size(), current_layer_mask.size()))

      if True:
        idx_of_concat = ((l - 1) - last_layer_last_encoder) / width
        if idx_of_concat.is_integer() and (idx_of_concat >= 0) and ((l - 1) < n_layers - 1 - width):
          concat_layer_idx = (l - 1) - width - idx_of_concat * 2 * width
          concat_layer_idx = int(concat_layer_idx)
          concatenate_layer_mask = keep_masks[concat_layer_idx * 2]
          concatenate_layer_out_c = concatenate_layer_mask.size(0)
          assert (concatenate_layer_out_c + front_layer_out_c == current_layer_in_c)
      else:
        if l == 8:
          concatenate_layer_mask = keep_masks[5 * 2]
        elif l == 10:
          concatenate_layer_mask = keep_masks[3 * 2]
        elif l == 12:
          concatenate_layer_mask = keep_masks[1 * 2]

        concatenate_layer_out_c = concatenate_layer_mask.size(0)
        assert (concatenate_layer_out_c + front_layer_out_c == current_layer_in_c)

    for idx_neuron in range(current_layer_in_c):
      if (concatenate_layer_out_c > 0) and (idx_neuron < concatenate_layer_out_c):
        if (current_layer_mask[:, idx_neuron, :, :, :].sum() == 0) \
                and (concatenate_layer_mask[idx_neuron, :, :, :, :].sum() != 0):
          invalid_area = concatenate_layer_mask[idx_neuron, :, :, :, :] == 1
          concatenate_layer_mask[idx_neuron, :, :, :, :][invalid_area] = 0
      else:
        idx_neuron_concat = idx_neuron - concatenate_layer_out_c
        if (current_layer_mask[:, idx_neuron, :, :, :].sum() == 0) \
                and (front_layer_mask[idx_neuron_concat, :, :, :, :].sum() != 0):
          invalid_area = front_layer_mask[idx_neuron_concat, :, :, :, :] == 1
          front_layer_mask[idx_neuron_concat, :, :, :, :][invalid_area] = 0

  # TODO
  # # Attention: Fill holes in neuron in_c, because after the above, in a neuron, in_c will be sparse
  # # but this neuron will be retained whenever there is ONE in_c is retained
  # for l in range(n_layers):
  #   weight_mask, bias_mask = keep_masks[2 * l], keep_masks[2 * l + 1]
  #   neuron_no = weight_mask.size(0)
  #   weight_sum = weight_mask.view(neuron_no, -1).sum(1)
  #   valid_neuron = weight_sum > 0
  #   invalid_neuron = weight_sum == 0
  #   weight_mask[valid_neuron] = 1
  #   weight_mask[invalid_neuron] = 0
  #
  #   if bias_mask is not None:
  #     bias_mask[valid_neuron] = 1
  #     bias_mask[invalid_neuron] = 0

  return keep_masks


def remove_redundant(keep_masks, prune_mode='param'):
  if keep_masks is None:
    return

  keep_masks = copy.deepcopy(keep_masks)

  if enable_verbose:
    print('=' * 20, 'remove redundant retains', '=' * 20)

  n_layers = len(keep_masks) // 2

  # Forward
  for l in range(n_layers - 1):
    weight_idx, bias_idx = 2 * l, 2 * l + 1
    next_weight_idx, next_bias_idx = 2 * (l + 1), 2 * (l + 1) + 1
    current_layer_mask = keep_masks[weight_idx]
    current_layer_bias_mask = keep_masks[bias_idx]
    next_layer_mask = keep_masks[next_weight_idx]
    next_layer_bias_mask = keep_masks[next_bias_idx]
    current_layer_mask = current_layer_mask.unsqueeze(2).unsqueeze(3) if (len(current_layer_mask.size()) == 2) else current_layer_mask
    next_layer_mask = next_layer_mask.unsqueeze(2).unsqueeze(3) if (len(next_layer_mask.size()) == 2) else next_layer_mask

    # for the case of flatten the output of convolutional layer, and connect
    # with a fully-connected layer, channels unmatched
    current_layer_out_c = current_layer_mask.size()[0]
    following_in_c = next_layer_mask.size()[1]
    if current_layer_out_c != following_in_c:
      if enable_verbose:
        print('Warning (this is fine), current layer: {}, following: {}'.format(current_layer_mask.size(), next_layer_mask.size()))

      next_layer_mask = next_layer_mask.view(-1, current_layer_out_c, following_in_c // current_layer_out_c, 1)

    for idx_neuron in range(current_layer_mask.size()[0]):
      if (current_layer_bias_mask is not None) and \
              ((current_layer_mask[idx_neuron, :, :, :].sum() + current_layer_bias_mask[idx_neuron] == 0)
               and (next_layer_mask[:, idx_neuron, :, :].sum() != 0)):
        exist_invalid = True
      elif (current_layer_bias_mask is None) and \
              ((current_layer_mask[idx_neuron, :, :, :].sum() == 0)
               and (next_layer_mask[:, idx_neuron, :, :].sum() != 0)):
        exist_invalid = True
      else:
        exist_invalid = False

      if exist_invalid:
        invalid_area = next_layer_mask[:, idx_neuron, :, :] == 1
        next_layer_mask[:, idx_neuron, :, :][invalid_area] = 0

    # Bug fixed, when enable_bias=True, mp and channel_prune results are different because when removing invalid retains in channel_prune,
    # bias should be removed when all of the channels of a neuron are removed, which is different from param_prune
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
    current_layer_mask = current_layer_mask.unsqueeze(2).unsqueeze(3) if (len(current_layer_mask.size()) == 2) else current_layer_mask
    front_layer_mask = front_layer_mask.unsqueeze(2).unsqueeze(3) if (len(front_layer_mask.size()) == 2) else front_layer_mask

    # for the case of flatten the output of convolutional layer, and connect
    # with a fully-connected layer, channels unmatched
    front_layer_out_c = front_layer_mask.size()[0]
    current_layer_in_c = current_layer_mask.size()[1]
    if current_layer_in_c != front_layer_out_c:
      if enable_verbose:
        print('Warnining (this is fine), front layer: {}, current layer: {}'.format(front_layer_mask.size(), current_layer_mask.size()))

      current_layer_mask = current_layer_mask.view(-1, front_layer_out_c, current_layer_in_c // front_layer_out_c, 1)

    for idx_neuron in range(current_layer_mask.size()[1]):
      if (front_layer_bias_mask is not None) and \
              ((current_layer_mask[:, idx_neuron, :, :].sum() == 0)
               and (front_layer_mask[idx_neuron, :, :, :].sum() + front_layer_bias_mask[idx_neuron] != 0)):
        exist_invalid = True
      elif (front_layer_bias_mask is None) and \
              ((current_layer_mask[:, idx_neuron, :, :].sum() == 0)
               and (front_layer_mask[idx_neuron, :, :, :].sum() != 0)):
        exist_invalid = True
      else:
        exist_invalid = False

      if exist_invalid:
        invalid_area = front_layer_mask[idx_neuron, :, :, :] == 1
        front_layer_mask[idx_neuron, :, :, :][invalid_area] = 0

        if (front_layer_bias_mask is not None) and (front_layer_bias_mask[idx_neuron] == 1):
          front_layer_bias_mask[idx_neuron] = 0

  return keep_masks


def dump_grad_mask(grads, mask, args):
  # Zhiwei dump this all_scores for message passing in MatLab
  params = np.zeros((len(grads),), dtype=np.object)
  masks = np.zeros((len(grads),), dtype=np.object)

  for idx, layer_param in enumerate(grads):
    if args.enable_raw:
      params[idx] = layer_param.cpu().numpy()
    else:
      params[idx] = (layer_param.abs()).cpu().numpy()

    masks[idx] = mask[idx].cpu().numpy()

  scio.savemat('../data/params_{}_{}.mat'.format(args.network, args.optimizer),
               {'params': params, 'masks': masks})


def param_prune(grads, param_sparsity, enable_norm=False):
  if enable_verbose:
    print('=' * 20, 'param prune (=SNIP)', '=' * 20)

  # Calculate threshold
  zero_fill = grads[0].new_zeros(1, dtype=torch.uint8)
  grad_vector = torch.cat([torch.flatten(grad) if (grad is not None) else zero_fill for grad in grads])
  norm_factor = grad_vector.sum() if enable_norm else 1
  grad_vector = grad_vector / norm_factor if enable_norm else grad_vector
  n_params = grad_vector.size()[0]
  threshold, _ = torch.topk(grad_vector, int(n_params * (1 - param_sparsity)), sorted=True)
  threshold_value = threshold[-1]

  # Prune on weights
  param_mask = [(grad / norm_factor >= threshold_value).float() if (grad is not None) else None for grad in grads]
  n_param_retained = torch.sum(torch.cat([torch.flatten(mask == 1) if (mask is not None) else zero_fill for mask in param_mask]))

  if enable_verbose:
    print('Weight prune, param sparsity:{}, keep {} out of {} weights ({:.4f}).' \
          .format(param_sparsity, n_param_retained, n_params, float(n_param_retained) / float(n_params)))

  return param_mask


def param_prune_3dunet(grads, param_sparsity, enable_norm=False):
  if enable_verbose:
    print('=' * 20, 'param prune (=SNIP)', '=' * 20)

  # Calculate threshold
  zero_fill = grads[0].new_zeros(1, dtype=torch.float)
  grad_vector = torch.cat([torch.flatten(grad) if (grad is not None) else zero_fill for grad in grads])
  norm_factor = grad_vector.sum() if enable_norm else 1
  grad_vector = grad_vector / norm_factor if enable_norm else grad_vector
  n_params = grad_vector.size()[0]
  threshold, _ = torch.topk(grad_vector, int(n_params * (1 - param_sparsity)), sorted=True)
  threshold_value = threshold[-1]

  # Prune on weights
  param_mask = [(grad / norm_factor >= threshold_value).float() if (grad is not None) else None for grad in grads]

  # Last conv all 1 for num of classes
  # if len(param_mask[-1]) == 1:
  if param_mask[-1] is not None and len(param_mask[-1]) > 0:
    param_mask[-1] = param_mask[-1].new_ones(param_mask[-1].size())

  if param_mask[-2] is not None and len(param_mask[-2]) > 0:
    param_mask[-2] = param_mask[-2].new_ones(param_mask[-2].size())

  n_param_retained = torch.sum(torch.cat([torch.flatten(mask == 1).float() if (mask is not None) else zero_fill for mask in param_mask]))

  if enable_verbose:
    print('Weight prune, param sparsity:{}, keep {} out of {} weights ({:.4f}).' \
          .format(param_sparsity, n_param_retained, n_params, float(n_param_retained) / float(n_params)))

  return param_mask


# Specific for 3dunet as only the last layer has bias and it has skip layers
def neuron_prune_3dunet(grads, neuron_sparsity, acc_mode,
                        layer_sparsity_list=None,
                        random_method=None,
                        random_sparsity=None,
                        random_sparsity_seed=0,
                        enable_layer_neuron_display=False,
                        resource_list_type=None,
                        resource_list=None,
                        resource_list_lambda=0):
  if enable_verbose:
    print('=' * 20, '3DUNet neuron prune', '=' * 20)

  neuron_grad_list = []
  n_layers = len(grads) // 2
  enable_dump_distribution = False

  if resource_list is not None:
    assert len(resource_list) == n_layers, \
      'n_layer from grad masks {} != n_layer from memory list {}'.format(n_layers, len(resource_list))

  # Get topk threshold
  for idx in range(n_layers):
    weight_idx, bias_idx = 2 * idx, 2 * idx + 1
    n_neuron_a_layer = grads[weight_idx].size(0)

    grad_weight = grads[weight_idx]
    grad_bias = grads[bias_idx] if (grads[bias_idx] is not None) else None

    # Only the last 3dconv has bias
    if acc_mode == 'sum':
      neuron_grad_accu = grad_weight.view(n_neuron_a_layer, -1).sum(dim=1)

      if grad_bias is not None:
        neuron_grad_accu = neuron_grad_accu + grad_bias
    elif acc_mode == 'mean':
      grads_a_layer = grad_weight.view(n_neuron_a_layer, -1)
      n_elements = grads_a_layer.size(1)

      if grad_bias is not None:
        neuron_grad_accu = (grads_a_layer.sum(1) + grad_bias) / (n_elements + 1)
      else:
        neuron_grad_accu = grads_a_layer.sum(1) / n_elements
    elif acc_mode == 'max':
      neuron_grad_accu, _ = grad_weight.view(n_neuron_a_layer, -1).max(dim=1)
    else:
      assert False

    neuron_grad_accu = neuron_grad_accu.abs()  # 24-Jan-2020 Neuron importance is abs()
    neuron_grad_list.append(neuron_grad_accu)

  neuron_grad_list_org, neuron_grad_list_grad, neuron_grad_list_grad_flops = [], [], []
  if enable_dump_distribution:
    neuron_grad_list_org = copy.deepcopy(neuron_grad_list)
    neuron_grad_list_org_mean = [neuron_grad.mean().cpu().numpy() for neuron_grad in neuron_grad_list_org]

  # TODO : Factor based on Neuron Importance or Memory List
  if random_method is None:
    if True:
      if resource_list_type.find('grad') > -1:
        neuron_grad_list_mean = torch.stack([neuron_grad.mean() for neuron_grad in neuron_grad_list], dim=0)
        neuron_grad_list_mean_max = neuron_grad_list_mean.max()
        neuron_grad_list_factor = neuron_grad_list_mean_max / neuron_grad_list_mean
        neuron_grad_list = [neuron_grad * neuron_grad_list_factor[idx] for idx, neuron_grad in enumerate(neuron_grad_list)]

        if enable_verbose:
          print('=> Layer factors based on grads: \n{}'.format(neuron_grad_list_factor.cpu().numpy()))

        if enable_dump_distribution:
          neuron_grad_list_grad = copy.deepcopy(neuron_grad_list)
          neuron_grad_list_grad_mean = [neuron_grad.mean().cpu().numpy() for neuron_grad in neuron_grad_list_grad]

      if any([resource_list_type.find(s) > -1 for s in ['flops', 'param', 'memory']]):
        resource_list_factor = F.softmax(-resource_list / resource_list.max(), dim=0)

        if True:
          neuron_grad_list = [neuron_grad * (1 + resource_list_lambda * resource_list_factor[idx]) for idx, neuron_grad in enumerate(neuron_grad_list)]
        else:
          neuron_grad_list = [neuron_grad * resource_list_factor[idx] for idx, neuron_grad in enumerate(neuron_grad_list)]

        if enable_verbose:
          print('=> Layer weights([0, 1]) based on resource: \n{}'.format(resource_list_factor.cpu().numpy()))

        if enable_dump_distribution:
          neuron_grad_list_grad_flops = copy.deepcopy(neuron_grad_list)
          neuron_grad_list_grad_flops_mean = [neuron_grad.mean().cpu().numpy() for neuron_grad in neuron_grad_list_grad_flops]
    else:
      if any([resource_list_type.find(s) > -1 for s in ['flops', 'param', 'memory']]):
        resource_list_factor = F.softmax(-resource_list / resource_list.max(), dim=0)
        neuron_grad_list_weighted = [neuron_grad * resource_list_factor[idx] for idx, neuron_grad in enumerate(neuron_grad_list)]

        neuron_grad_list_weighted_mean = torch.stack([neuron_grad.mean() for neuron_grad in neuron_grad_list_weighted], dim=0)
        neuron_grad_list_weighted_mean_max = neuron_grad_list_weighted_mean.max()
        neuron_grad_list_factor = neuron_grad_list_weighted_mean_max / neuron_grad_list_weighted_mean
        neuron_grad_list = [neuron_grad * neuron_grad_list_factor[idx] for idx, neuron_grad in enumerate(neuron_grad_list)]

  # Get weight mask
  param_mask =[]

  if (layer_sparsity_list is not None) and (layer_sparsity_list > 0):  # Layer-wise neuron pruning
    enable_layer_sparsity_list = isinstance(layer_sparsity_list, list)
    assert (len(layer_sparsity_list) == n_layers) if enable_layer_sparsity_list else True
    n_neurons = 0

    for idx in range(n_layers):
      weight_idx, bias_idx = 2 * idx, 2 * idx + 1
      weight_mask_a_layer = grads[weight_idx].new_zeros(grads[weight_idx].size())

      neuron_grad_vector = neuron_grad_list[idx]
      n_neurons_a_layer = neuron_grad_vector.size(0)

      if idx == n_layers - 1:  # last layer
        layer_sparsity = 0
      else:
        layer_sparsity = layer_sparsity_list[idx] if enable_layer_sparsity_list else layer_sparsity_list

      threshold, _ = torch.topk(neuron_grad_vector, int(np.ceil(n_neurons_a_layer * (1 - layer_sparsity))), sorted=True)
      threshold_value = threshold[-1]

      if enable_verbose:
        print('===> Layer-wise neuron pruning, layer: {}, neurons: {}, retained: {}' \
              .format(idx, n_neurons_a_layer, int(np.ceil(n_neurons_a_layer * (1 - layer_sparsity)))))

      n_neurons += n_neurons_a_layer
      retained_area = neuron_grad_vector >= threshold_value  # neuron indices
      weight_mask_a_layer[retained_area] = 1  # retained_area refers to the first dimension
      param_mask.append(weight_mask_a_layer)

      if grads[bias_idx] is not None:
        bias_mask_a_layer = grads[bias_idx].new_zeros(grads[bias_idx].size())
        bias_mask_a_layer[retained_area] = 1
        param_mask.append(bias_mask_a_layer)
      else:
        param_mask.append(None)
  elif random_method is not None:  # Random pruning
    neuron_grad_vector = torch.cat(neuron_grad_list, dim=0)
    n_neurons = neuron_grad_vector.size(0)

    torch.manual_seed(random_sparsity_seed)
    torch.cuda.manual_seed(random_sparsity_seed)
    np.random.seed(random_sparsity_seed)

    if random_method == 0:
      random_retain_mask = torch.zeros(n_neurons, dtype=torch.uint8)
      indices = np.arange(n_neurons)
      # np.random.shuffle(indices)
      choice = torch.from_numpy(np.random.choice(indices, int(n_neurons * (1 - random_sparsity)), replace=False))
      random_retain_mask[choice] = 1
    elif random_method == 1:
      random_retain_mask = torch.ones(n_neurons, dtype=torch.uint8)
      indices = np.arange(n_neurons)
      # np.random.shuffle(indices)
      choice = torch.from_numpy(np.random.choice(indices, int(n_neurons * random_sparsity), replace=False))
      random_retain_mask[choice] = 0
    else:
      assert False, 'Invalid random method: {}'.format(random_method)
    extract_start, extract_end = 0, 0

    for idx in range(n_layers):
      weight_idx, bias_idx = 2 * idx, 2 * idx + 1
      weight_mask_a_layer = grads[weight_idx].new_zeros(grads[weight_idx].size())
      n_neuron_a_layer = grads[weight_idx].size(0)
      extract_end += n_neuron_a_layer

      if idx == n_layers - 1:  # last layer
        retained_area = neuron_grad_list[idx] >= 0
      else:
        retained_area = random_retain_mask[extract_start : extract_end]

      weight_mask_a_layer[retained_area] = 1
      param_mask.append(weight_mask_a_layer)

      if grads[bias_idx] is not None:
        bias_mask_a_layer = grads[bias_idx].new_zeros(grads[bias_idx].size())
        bias_mask_a_layer[retained_area] = 1
        param_mask.append(bias_mask_a_layer)
      else:
        param_mask.append(None)

      extract_start = extract_end
  else:  # Network neuron pruning
    neuron_grad_vector = torch.cat(neuron_grad_list, dim=0)

    n_neurons = neuron_grad_vector.size(0)
    threshold, _ = torch.topk(neuron_grad_vector, int(n_neurons * (1 - neuron_sparsity)), sorted=True)
    threshold_value = threshold[-1]

    for idx in range(n_layers):
      if idx == n_layers - 1:  # last layer
        threshold_value_new = -np.inf
      else:
        threshold_value_new = threshold_value

      weight_idx, bias_idx = 2 * idx, 2 * idx + 1
      weight_mask_a_layer = grads[weight_idx].new_zeros(grads[weight_idx].size())
      retained_area = neuron_grad_list[idx] >= threshold_value_new  # neuron indices
      weight_mask_a_layer[retained_area] = 1  # retained_area refers to the first dimension
      param_mask.append(weight_mask_a_layer)

      if grads[bias_idx] is not None:
        bias_mask_a_layer = grads[bias_idx].new_zeros(grads[bias_idx].size())
        bias_mask_a_layer[retained_area] = 1
        param_mask.append(bias_mask_a_layer)
      else:
        param_mask.append(None)

  if enable_layer_neuron_display:
    for idx in range(n_layers):
      n_neurons_a_layer = param_mask[2 * idx].size(0)
      n_neuron_retained_a_layer = (param_mask[2 * idx].view(n_neurons_a_layer, -1).sum(1) > 0).sum()

      if enable_verbose:
        print('Conv layer id: {}, neuron retained: {}/{} ({:.2f}%), size: {}' \
              .format(idx, n_neuron_retained_a_layer,
                      n_neurons_a_layer,
                      float(n_neuron_retained_a_layer) * 100 / float(n_neurons_a_layer),
                      param_mask[2 * idx].size()))

  zero_fill = grads[0].new_zeros(1, dtype=torch.uint8)
  n_neuron_retained = torch.sum(torch.cat([torch.flatten(mask.view(mask.size()[0], -1)[:, 0] == 1) \
                                             if (idx % 2 == 0) else zero_fill for idx, mask in enumerate(param_mask)]))
  n_param_retained = torch.sum(torch.cat([torch.flatten(mask == 1) \
                                            if (mask is not None) else zero_fill for mask in param_mask]))
  n_params = torch.sum(torch.cat([torch.flatten(mask >= 0) \
                                    if (mask is not None) else zero_fill for mask in param_mask]))

  if enable_verbose:
    print('Neuron prune, neuron sparsity:{}, keep {} out of {} neurons ({:.4f}), keep {} out of {} weights ({:.4f}).' \
          .format(neuron_sparsity, n_neuron_retained, n_neurons, float(n_neuron_retained) / float(n_neurons),
                  n_param_retained, n_params, float(n_param_retained) / float(n_params)))

  if enable_dump_distribution:
    neuron_mask = []
    for idx, mask in enumerate(param_mask):
      if idx % 2 == 0:
        n_neurons = mask.size(0)
        value = mask.view(n_neurons, -1).sum(1) > 0
        neuron_mask.append(value)

    scio.savemat('dump/neuron_list.mat', {'org': torch.cat(neuron_grad_list_org).cpu().numpy(),
                                          'org_mean': neuron_grad_list_org_mean,
                                          'grad': torch.cat(neuron_grad_list_grad).cpu().numpy(),
                                          'grad_mean': neuron_grad_list_grad_mean,
                                          'grad_flops': torch.cat(neuron_grad_list_grad_flops).cpu().numpy(),
                                          'grad_flops_mean': neuron_grad_list_grad_flops_mean,
                                          'mask': torch.cat(neuron_mask).cpu().numpy(),
                                          'number': [neuron_grad.size(0) for neuron_grad in neuron_grad_list_org]})

  return param_mask


# Prune neuron is not good, may prune a whole layer
def neuron_prune(grads, neuron_sparsity, acc_mode):
  if enable_verbose:
    print('=' * 20, 'neuron prune', '=' * 20)

  neuron_grad_list = []
  n_layers = len(grads) // 2

  # Get topk threshold
  for idx in range(n_layers):
    weight_idx, bias_idx = 2 * idx, 2 * idx + 1
    n_neuron_a_layer = grads[weight_idx].size()[0]

    grad_weight = grads[weight_idx]
    grad_bias = grads[bias_idx] if (grads[bias_idx] is not None) else None

    if acc_mode == 'sum':
      neuron_grad_accu = grad_weight.view(n_neuron_a_layer, -1).sum(dim=1)
      neuron_grad_accu = neuron_grad_accu + grad_bias if (grad_bias is not None) else neuron_grad_accu
    elif acc_mode == 'mean':
      grads_a_layer = grad_weight.view(n_neuron_a_layer, -1)
      n_elements = grads_a_layer.size(1)

      if grad_bias is not None:
        neuron_grad_accu = (grads_a_layer.sum(1) + grad_bias) / (n_elements + 1)
      else:
        neuron_grad_accu = grads_a_layer.sum(1) / n_elements
    elif acc_mode == 'max':
      neuron_grad_accu, _ = grad_weight.view(n_neuron_a_layer, -1).max(dim=1)
    else:
      assert False

    neuron_grad_list.append(neuron_grad_accu)

  neuron_grad_vector = torch.cat(neuron_grad_list, dim=0)
  n_neurons = neuron_grad_vector.size()[0]
  threshold, _ = torch.topk(neuron_grad_vector, int(n_neurons * (1 - neuron_sparsity)), sorted=True)
  threshold_value = threshold[-1]

  # Get weight mask
  param_mask = []
  for idx in range(n_layers):
    weight_idx, bias_idx = 2 * idx, 2 * idx + 1
    weight_mask_a_layer = grads[weight_idx].new_zeros(grads[weight_idx].size())
    retained_area = neuron_grad_list[idx] >= threshold_value  # neuron indices
    weight_mask_a_layer[retained_area] = 1  # retained_area refers to the first dimension
    param_mask.append(weight_mask_a_layer)

    if grads[bias_idx] is not None:
      bias_mask_a_layer = grads[bias_idx].new_zeros(grads[bias_idx].size())
      bias_mask_a_layer[retained_area] = 1
      param_mask.append(bias_mask_a_layer)
    else:
      param_mask.append(None)

  zero_fill = param_mask[0].new_zeros(1, dtype=torch.uint8)
  n_neuron_retained = torch.sum(torch.cat([torch.flatten(mask.view(mask.size()[0], -1)[:, 0] == 1) if (idx % 2 == 0) else zero_fill for idx, mask in enumerate(param_mask)]))
  n_param_retained = torch.sum(torch.cat([torch.flatten(mask == 1) if (mask is not None) else zero_fill for mask in param_mask]))
  n_params = torch.sum(torch.cat([torch.flatten(mask >= 0) if (mask is not None) else zero_fill for mask in param_mask]))

  if enable_verbose:
    print('Neuron prune, neuron sparsity:{}, keep {} out of {} neurons ({:.4f}), keep {} out of {} weights ({:.4f}).' \
          .format(neuron_sparsity, n_neuron_retained, n_neurons, float(n_neuron_retained) / float(n_neurons),
                  n_param_retained, n_params, float(n_param_retained) / float(n_params)))

  return param_mask


def channel_prune(grads, channel_sparsity, acc_mode='max', norm='max'):
  if enable_verbose:
    print('=' * 20, 'channel prune', '=' * 20)

  grads_org = copy.deepcopy(grads)
  grads = copy.deepcopy(grads)
  n_layers = len(grads) // 2
  grads = convert_dim_conv2fully(grads)
  channel_accum_grad_list, threshold_value = cal_channel_prune_grad(grads, channel_sparsity, mode=acc_mode, norm=norm)

  # Prune on channel
  param_mask = []
  for idx in range(n_layers):
    weight_idx, bias_idx = 2 * idx, 2 * idx + 1
    weight_mask_a_layer = grads[weight_idx].new_zeros(grads[weight_idx].size())
    retained_area = channel_accum_grad_list[idx] >= threshold_value
    weight_mask_a_layer[retained_area] = 1  # retained_area refers to the first two dimensions
    param_mask.append(weight_mask_a_layer)

    if grads[bias_idx] is not None:
      bias_mask_a_layer = grads[bias_idx].new_zeros(grads[bias_idx].size())
      n_neurons = bias_mask_a_layer.size()[0]
      retained_area = weight_mask_a_layer.view(n_neurons, -1).sum(dim=1) > 0
      bias_mask_a_layer[retained_area] = 1
      param_mask.append(bias_mask_a_layer)
    else:
      param_mask.append(None)

  zero_fill = param_mask[0].new_zeros(1, dtype=torch.uint8)
  n_weight_retained = torch.sum(torch.cat([torch.flatten(mask == 1) if (mask is not None) else zero_fill for mask in param_mask]))
  n_weights = torch.sum(torch.cat([torch.flatten(mask >= 0) if (mask is not None) else zero_fill for mask in param_mask]))

  if enable_verbose:
    print('Channel prune, channel sparsity:{}, keep {} out of {} weights ({:.4f})' \
          .format(channel_sparsity, n_weight_retained, n_weights, float(n_weight_retained) / float(n_weights)))

  param_mask = resume_dim_conv2fully(param_mask, grads_org)

  return param_mask


def hidden_layer_prune(grads, sparsity, enable_sum=False):
  if enable_verbose:
    print('=' * 20, 'hidden layer prune', '=' * 20)

  hidden_grad_list = []
  hidden_masks = []

  # Get topk threshold
  if enable_sum:
    for grad in grads:
      hidden_grad_list.append(grad.sum(0))  # one layer one 3D mask no matter the out_channels
  else:
    hidden_grad_list = grads

  hidden_grad_vector = torch.cat([data.flatten() for data in hidden_grad_list], dim=0)
  n_elements = hidden_grad_vector.size()[0]
  threshold, _ = torch.topk(hidden_grad_vector, int(n_elements * (1 - sparsity)), sorted=True)
  threshold_value = threshold[-1]

  for hidden_grad in hidden_grad_list:
    hidden_mask = hidden_grad.new_zeros(hidden_grad.size())
    hidden_mask[hidden_grad >= threshold_value] = 1
    hidden_masks.append(hidden_mask.unsqueeze(0))  # for batch

  return hidden_masks


def pruning(file_name,
            model,
            train_dataloader,
            criterion,
            args,
            enable_3dunet=False,
            enable_hidden_sum=False,
            width=2,
            resource_list=None,
            network_name='3dunet'):
  enable_kernel_mask = (args.enable_neuron_prune or args.enable_param_prune)
  enable_hidden_mask = args.enable_hidden_layer_prune

  # ==== Get gradients
  if (file_name is not None) and os.path.exists(file_name):
    obj = np.load(file_name, allow_pickle=True)
    kernel_grads_abs_average = obj.item()['kernel_mask_grad'] if enable_kernel_mask else []
    hidden_grads_abs_average = obj.item()['hidden_mask_grad'] if enable_hidden_mask else []
  else:
    print(args)
    model = copy.deepcopy(model)
    model = add_mask_for_grad(model, args,
                              enable_kernel_mask=enable_kernel_mask,
                              enable_hidden_mask=enable_hidden_mask)

    if enable_hidden_mask:  # preset hidden mask size by a fixed input size
      randint_input = torch.ones(args.batch, 1, args.spatial_size,
                                 args.spatial_size, args.spatial_size,
                                 dtype=torch.float).to(args.device)
      model.forward(randint_input)
      remove_hooks(model)

    kernel_grads_abs_average, hidden_grads_abs_average = [], []
    batch_total = len(train_dataloader)
    time_start = time.time()

    for idx, data in enumerate(train_dataloader):
      if ((batch_total > 1000) and idx % 100 == 0) or (batch_total <= 1000):
        print('Pruning, batch: {} / {}'.format(idx + 1, batch_total))

      if True:
        model.zero_grad()  # original snip pytorch code due to learnable mask that is not in optimizer
      else:
        optimizer.zero_grad()  # this is regular one that all learnable parameters are set into optimizer

      # For stereo and otherwise
      if args.dataset in {'sceneflow'}:
        imgL, imgR, disp_L = data

        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))
        disp_L = Variable(torch.FloatTensor(disp_L))
        
        if args.enable_cuda:
          imgL, imgR, disp_L = imgL.cuda(), imgR.cuda(), disp_L.cuda()
        
        mask = disp_L < args.maxdisp
        mask.detach_()

        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_L[mask], reduction='mean') \
               + 0.7 * F.smooth_l1_loss(output2[mask], disp_L[mask], reduction='mean') \
               + F.smooth_l1_loss(output3[mask], disp_L[mask], reduction='mean')
      else:
        if isinstance(data, list):
          input, gt = data[0], data[1]
        else:
          input, gt = data['x'], data['y']

        actual_batch = input.size(0)

        if args.enable_cuda:
          input, gt = input.cuda(), gt.cuda()

        prediction = model(input)
        prediction = prediction.reshape(actual_batch, args.n_class, -1)
        gt = gt.reshape(actual_batch, -1)
        loss = criterion(prediction, gt)

      loss.backward()

      kernel_grad_abs, hidden_grad_abs = get_mask_grad(model,
                                                       enable_kernel_mask=enable_kernel_mask,
                                                       enable_hidden_mask=enable_hidden_mask,
                                                       enable_raw_grad=args.enable_raw_grad)

      if enable_kernel_mask:
        kernel_grads_abs_average = update_grads_average(kernel_grads_abs_average, kernel_grad_abs, idx)

      if enable_hidden_mask:
        hidden_grads_abs_average = update_grads_average(hidden_grads_abs_average, hidden_grad_abs, idx)

      # torch.cuda.empty_cache()  # too slow

    duration_pruning = time.time() - time_start
    if (file_name is not None):
      np.save(file_name, {'kernel_mask_grad': kernel_grads_abs_average,
                          'hidden_mask_grad': hidden_grads_abs_average,
                          'time': duration_pruning})

  # ==== Get kernel mask by pruning on kernels, including removing redundent
  if enable_kernel_mask:
    if args.enable_param_prune:
      if False:  # previous one
        if enable_3dunet or (network_name == '3dunet'):
          kernel_mask = param_prune_3dunet(kernel_grads_abs_average, args.param_sparsity, enable_norm=False)
          kernel_mask_clean = remove_redundant_3dunet(kernel_mask, width=width)
        else:
          assert False
      else:
        kernel_mask = param_prune_3dunet(kernel_grads_abs_average, args.param_sparsity, enable_norm=False)

        if network_name == '3dunet':
          kernel_mask_clean = remove_redundant_3dunet(kernel_mask, width=width)
        elif network_name == 'mobilenetv2':
          kernel_mask_clean = remove_redundant_3dmobilenet(kernel_mask)
        elif network_name == 'i3d':
          kernel_mask_clean = remove_redundant_I3D(kernel_mask)
        elif network_name == 'psm':
          valid_neuron_list_clean = remove_redundant_PSM(kernel_mask, mode=args.PSM_mode)
        else:
          assert False
    elif args.enable_neuron_prune:
      if enable_3dunet or (network_name == '3dunet'):
        kernel_mask = neuron_prune_3dunet(kernel_grads_abs_average,
                                          args.neuron_sparsity,
                                          args.acc_mode,
                                          layer_sparsity_list=args.layer_sparsity_list,
                                          random_method=args.random_method,
                                          random_sparsity=args.random_sparsity,
                                          random_sparsity_seed=args.random_sparsity_seed,
                                          enable_layer_neuron_display=args.enable_layer_neuron_display,
                                          resource_list_type=args.resource_list_type,
                                          resource_list=resource_list,
                                          resource_list_lambda=args.resource_list_lambda)
        kernel_mask_clean = remove_redundant_3dunet(kernel_mask, width=width)
      elif network_name == 'mobilenetv2':
        kernel_mask = neuron_prune_3dunet(kernel_grads_abs_average,
                                          args.neuron_sparsity,
                                          args.acc_mode,
                                          layer_sparsity_list=args.layer_sparsity_list,
                                          random_method=args.random_method,
                                          random_sparsity=args.random_sparsity,
                                          random_sparsity_seed=args.random_sparsity_seed,
                                          resource_list_type=args.resource_list_type,
                                          resource_list=resource_list,
                                          resource_list_lambda=args.resource_list_lambda)
        kernel_mask_clean = remove_redundant_3dmobilenet(kernel_mask)
      elif network_name == 'i3d':
        kernel_mask = neuron_prune_3dunet(kernel_grads_abs_average,
                                          args.neuron_sparsity,
                                          args.acc_mode,
                                          layer_sparsity_list=args.layer_sparsity_list,
                                          random_method=args.random_method,
                                          random_sparsity=args.random_sparsity,
                                          random_sparsity_seed=args.random_sparsity_seed,
                                          resource_list_type=args.resource_list_type,
                                          resource_list=resource_list,
                                          resource_list_lambda=args.resource_list_lambda,
                                          enable_layer_neuron_display=False)
        kernel_mask_clean = remove_redundant_I3D(kernel_mask)
      elif network_name == 'psm':
        kernel_mask = neuron_prune_3dunet(kernel_grads_abs_average,
                                          args.neuron_sparsity,
                                          args.acc_mode,
                                          layer_sparsity_list=args.layer_sparsity_list,
                                          random_method=args.random_method,
                                          random_sparsity=args.random_sparsity,
                                          random_sparsity_seed=args.random_sparsity_seed,
                                          resource_list_type=args.resource_list_type,
                                          resource_list=resource_list,
                                          resource_list_lambda=args.resource_list_lambda,
                                          enable_layer_neuron_display=False)
        valid_neuron_list_clean = remove_redundant_PSM(kernel_mask, mode=args.PSM_mode)
      else:
        kernel_mask = neuron_prune(kernel_grads_abs_average,
                                   args.neuron_sparsity, args.acc_mode)
        kernel_mask_clean = remove_redundant(kernel_mask)

    if network_name == 'psm':
      for idx, valid_neuron in enumerate(valid_neuron_list_clean):  # previously use kernel_mask, but no difference I think
        if valid_neuron[0] == 0:
          print('All removed at {}th layer of valid_neuron_list'.format(idx // 2))
          status = -1
          return [status]
    else:
      do_statistics(kernel_mask, kernel_mask_clean)

      for idx, mask in enumerate(kernel_mask_clean):  # previously use kernel_mask, but no difference I think
        if (mask is not None) and (mask.sum() == 0):
          print('All removed at {}th layer of kernel_mask_clean'.format(idx // 2))
          status = -1
          return [status]
  else:
    kernel_mask_clean = None

  # ==== Get hidden layer mask by pruning on hidden layers
  if enable_hidden_mask:
    hidden_masks = hidden_layer_prune(hidden_grads_abs_average, args.hidden_layer_sparsity,
                                      enable_sum=enable_hidden_sum)

    n_elements_raw_total, n_elements_raw_retain = 0, 0
    n_elements_expand_total, n_elements_expand_retain = 0, 0

    for idx, (hidden_mask, hidden_grad) in enumerate(zip(hidden_masks, hidden_grads_abs_average)):
      # Expanded to out_channels
      out_channels = hidden_grad.size(0)
      n_elements_expand_total += np.double((hidden_grad >= 0).sum().cpu().numpy())

      if enable_hidden_sum:
        n_elements_expand_retain += np.double(out_channels * (hidden_mask > 0).sum().cpu().numpy())
      else:
        n_elements_expand_retain += np.double((hidden_mask > 0).sum().cpu().numpy())

      # Raw this will be the same as preset sparsity
      n_elements_raw_total += np.double((hidden_mask >= 0).sum().cpu().numpy())
      n_elements_raw_retain += np.double((hidden_mask > 0).sum().cpu().numpy())

    if enable_verbose:
      print('Hidden layer pruning, preset: {:.2f}%, enable hidden sum: {};\n'
            'raw retain {}/{} ({:.2f}%);\n'
            'expand retain {}({:.2f}MB)/{}({:.2f}MB) ({:.4f}%) for ONE batch'. \
            format((1 - args.hidden_layer_sparsity) * 100, enable_hidden_sum,
                   n_elements_raw_retain, n_elements_raw_total,
                   n_elements_raw_retain * 100 / n_elements_raw_total,
                   n_elements_expand_retain, n_elements_expand_retain * 4 / 1024**2,
                   n_elements_expand_total, n_elements_expand_total * 4 / 1024**2,
                   n_elements_expand_retain * 100 / n_elements_expand_total))
  else:
    hidden_masks = None

  status = 0

  if network_name == 'psm':
    return status, valid_neuron_list_clean, hidden_masks
  else:
    return status, kernel_mask_clean, hidden_masks


if __name__ == '__main__':
  torch.manual_seed(2019)
  torch.cuda.manual_seed_all(2019)

  # Test convert_dim_conv2fully
  conv_grads = []
  conv_grads.append(torch.randn(2, 3, 3, 3, dtype=torch.float32))
  conv_grads.append(torch.randn(10, 8, 1, 1, dtype=torch.float32))
  conv_grads_new = convert_dim_conv2fully(conv_grads)
  conv_grads_resume = resume_dim_conv2fully(conv_grads_new, conv_grads)
  print('Size, original:{}, convert:{}, resume:{}' \
        .format(conv_grads[1].size(), conv_grads_new[1].size(), conv_grads_resume[1].size()))

  # Test pure convolutional layers
  conv_grads = []
  conv_grads.append(torch.tensor([[[[2]], [[3]], [[1]]],
                                  [[[0]], [[7]], [[2]]]], dtype=torch.float32))
  conv_grads.append(torch.tensor([2, 7], dtype=torch.float32))
  conv_grads.append(torch.tensor([[[[3], [0], [2]], [[1], [8], [0]]],
                                  [[[0], [5], [2]], [[1], [2], [3]]],
                                  [[[4], [7], [6]], [[3], [8], [3]]]], dtype=torch.float32))
  conv_grads.append(torch.tensor([3, 5, 8], dtype=torch.float32))
  conv_grads.append(torch.tensor([[[[2], [3]], [[7], [1]], [[2], [8]]],
                                  [[[2], [2]], [[1], [0]], [[3], [7]]]], dtype=torch.float32))
  conv_grads.append(torch.tensor([6, 0], dtype=torch.float32))

  param_mask = param_prune(conv_grads, param_sparsity=0.8)
  param_mask_clean = remove_redundant(param_mask)
  invalid_a_layer, retained_a_layer, all_a_layer = do_statistics(param_mask, param_mask_clean)
  assert torch.equal(invalid_a_layer.int(), torch.tensor([0, 2, 1], dtype=torch.int))
  assert torch.equal(retained_a_layer.int(), torch.tensor([2, 4, 3], dtype=torch.int))
  assert torch.equal(all_a_layer.int(), torch.tensor([8, 21, 14], dtype=torch.int))

  channel_mask = channel_prune(conv_grads, channel_sparsity=0.7, acc_mode='mean')
  channel_mask_clean = remove_redundant(channel_mask, prune_mode='channel')
  invalid_a_layer, retained_a_layer, all_a_layer = do_statistics(channel_mask, channel_mask_clean)
  assert torch.equal(invalid_a_layer.int(), torch.tensor([0, 3, 2], dtype=torch.int))
  assert torch.equal(retained_a_layer.int(), torch.tensor([2, 7, 5], dtype=torch.int))
  assert torch.equal(all_a_layer.int(), torch.tensor([8, 21, 14], dtype=torch.int))

  channel_mask = channel_prune(conv_grads, channel_sparsity=0.7, acc_mode='max', norm='max')
  channel_mask_clean = remove_redundant(channel_mask, prune_mode='channel')
  do_statistics(channel_mask, channel_mask_clean)

  mp_mask = message_passing_prune(conv_grads, channel_sparsity=0.7, penalty=10, accu_mode='max', norm='max')

  for idx in range(len(channel_mask_clean)):
    print(idx, channel_mask_clean[idx].flatten(), mp_mask[idx].flatten())
