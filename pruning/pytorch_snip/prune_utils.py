import torch
import copy


def convert_dim_conv2fully(grads):
  grads = copy.deepcopy(grads)  # grads is a reference, changing its size in function will change it externally

  n_layers = len(grads) // 2

  for idx in range(n_layers - 1):
    weight_idx = 2 * idx
    next_weight_idx = 2 * (idx + 1)
    current_layer = grads[weight_idx]
    next_layer = grads[next_weight_idx]
    out_c_current = current_layer.size()[0]
    next_layer_size_len = len(next_layer.size())

    if next_layer_size_len == 4:
      out_c_next, in_c_next, h_next, w_next = next_layer.size()
    elif next_layer_size_len == 2:
      out_c_next, in_c_next = next_layer.size()
      h_next, w_next = 1, 1
    else:
      assert False

    # This usually happens from a convoluational layer to a fully-connected layer
    # because for some network, the output of a convolutional layer will be flatten, then to a fully-connected layer,
    # such as lenet5 and lenet5_caffe
    if out_c_current != in_c_next:
      assert (h_next == 1) and (w_next == 1)
      grads[next_weight_idx] = next_layer.view(out_c_next, out_c_current, (in_c_next // out_c_current) * h_next, w_next)

  return grads


def resume_dim_conv2fully(mask, grads):
  mask = copy.deepcopy(mask)  # grads is a reference, changing its size in function will change it externally

  assert len(mask) == len(grads)
  n_layers = len(grads) // 2

  for idx in range(n_layers):
    weight_idx = 2 * idx
    mask_current = mask[weight_idx]
    grad_current = grads[weight_idx]
    if mask_current.size() != grad_current.size():
      assert mask_current.flatten().size() == grad_current.flatten().size()
      mask[weight_idx] = mask_current.view(grad_current.size())

  return mask


def check_same(input_a, input_b):
  if (input_a is None) or (input_b is None):
    return False

  is_same = True

  if isinstance(input_a, list):
    assert len(input_a) == len(input_b)
    num = len(input_a)

    for idx in range(num):
      if not torch.equal(input_a[idx], input_b[idx]):
        is_same = False
        break
  else:
    is_same = False if (not torch.equal(input_a, input_b)) else True

  return is_same


def cal_channel_prune_grad(grads,  channel_sparsity, mode='max', norm='max'):
  n_layers = len(grads) // 2
  channel_accum_grad_list = []

  for idx in range(n_layers):
    weight_idx, bias_idx = 2 * idx, 2 * idx + 1
    grad_size = grads[weight_idx].size()
    out_c, in_c = grad_size[0], grad_size[1]

    # Bug: how to define the importance of a channel:
    # 'sum' not good, fully-connected layers would be removed dramatically as its kernel size is just one, would have 0-retained layer
    # 'mean', not good, convolutional layers would be removed dramatically as its kernel size is much larger than fully-connected layers
    #                   (whose kernel size is 1), the importance of a channel will be decreased by average.
    # 'max', good, highest grad decides how important this channel is
    if mode == 'sum':
      channel_accum = grads[weight_idx].view(out_c, in_c, -1).sum(2)
      channel_accum = channel_accum + grads[bias_idx].view(out_c, 1).repeat(1, in_c) if (grads[bias_idx] is not None) else channel_accum
    elif mode == 'mean':
      if grads[bias_idx] is not None:
        grads_a_layer = grads[weight_idx].view(out_c, in_c, -1)
        n_elements = grads_a_layer.size()[-1]
        channel_accum = grads_a_layer.sum(2)
        channel_accum = (channel_accum + grads[bias_idx].view(out_c, 1)) / (n_elements + 1)
      else:
        channel_accum = grads[weight_idx].view(out_c, in_c, -1).mean(2)
    elif mode == 'max':
      grads_a_layer = grads[weight_idx].view(out_c, in_c, -1)
      channel_accum, _ = grads_a_layer.max(2)
    else:
      assert False

    channel_accum_grad_list.append(channel_accum)

  # Calculate threshold
  channel_amu_grad_flatten = torch.cat([channel_accum_grad_list[idx].flatten() for idx in range(n_layers)], dim=0)
  n_channels = channel_amu_grad_flatten.size()[0]
  threshold, _ = torch.topk(channel_amu_grad_flatten, int(n_channels * (1 - channel_sparsity)), sorted=True)
  threshold = threshold[-1]

  if norm == 'max':
    norm_factor = channel_amu_grad_flatten.max()
  elif norm == 'sum':
    norm_factor = channel_amu_grad_flatten.sum()
  else:
    norm_factor = 1

  for idx in range(n_layers):
    channel_accum_grad_list[idx] /= norm_factor

  threshold /= norm_factor

  return channel_accum_grad_list, threshold