import copy
import sys
sys.path.append('./pruning/pytorch_snip')
from prune_utils import cal_channel_prune_grad, convert_dim_conv2fully, resume_dim_conv2fully


# This is to fix a bug to set label 1 when two costs are equal for a channel
# see all modifications by searching "argmin"
def find_equal_area(binary_cost):
  equal_area = binary_cost[:, :, 0] == binary_cost[:, :, 1]
  return equal_area


def message_passing_prune(grads, channel_sparsity, penalty, accu_mode='max', norm='max'):
  print('=' * 20, 'message passing prune', '=' * 20)
  n_layers = len(grads) // 2
  grads_org = copy.deepcopy(grads)
  grads = convert_dim_conv2fully(grads)
  accum_grads, threshold = cal_channel_prune_grad(grads, channel_sparsity, mode=accu_mode, norm=norm)

  # Set unary
  unary = []
  for idx in range(n_layers):
    cout, cin = accum_grads[idx].size()
    unary_a_layer = accum_grads[idx].new_zeros(cout, cin, 2)
    unary_a_layer[:, :, 1] = -(accum_grads[idx] - threshold)
    unary.append(unary_a_layer)

  # Initialize message
  message_forward, message_backward = [], []
  for idx in range(n_layers):
    cout, cin = accum_grads[idx].size()
    message_forward.append(accum_grads[idx].new_zeros(cout, cin, 2))
    message_backward.append(accum_grads[idx].new_zeros(cout, cin, 2))

  # Forward pass
  for idx in range(1, n_layers, 1):
    weight_idx = idx
    front_weight_idx = idx - 1
    front_layer_cost = unary[front_weight_idx] + message_forward[front_weight_idx]
    front_layer_channel_label = front_layer_cost.argmin(dim=2)
    front_layer_channel_label[find_equal_area(front_layer_cost)] = 1
    front_layer_neuron_label = 1 - (1 - front_layer_channel_label).prod(dim=1)
    front_layer_num_neuron = front_layer_neuron_label.size(0)

    for n_idx in range(front_layer_num_neuron):
      message_forward[weight_idx][:, n_idx, 0] = 0
      message_forward[weight_idx][:, n_idx, 1] = penalty * (1 - front_layer_neuron_label[n_idx])

  # Backward pass
  for idx in range(n_layers - 2, -1, -1):
    weight_idx = idx
    next_weight_idx = idx + 1
    next_layer_cost = unary[next_weight_idx] + message_backward[next_weight_idx]
    current_layer_num_neuron = unary[weight_idx].size(0)
    next_layer_status = next_layer_cost.argmin(dim=2)
    next_layer_status[find_equal_area(next_layer_cost)] = 1
    next_layer_channel_cross_label = 1 - (1 - next_layer_status).prod(dim=0)

    for n_idx in range(current_layer_num_neuron):
      message_backward[weight_idx][n_idx, :, 0] = 0
      message_backward[weight_idx][n_idx, :, 1] = penalty * (1 - next_layer_channel_cross_label[n_idx])

  # Final
  mask = []
  for idx in range(len(grads)):
    if grads[idx] is not None:
      mask.append(grads[idx].new_full(grads[idx].size(), -1))
    else:
      mask.append(None)

  for idx in range(n_layers):
    weight_idx, bias_idx = 2 * idx, 2 * idx + 1
    cost = unary[idx] + message_forward[idx] + message_backward[idx]
    channel_label = cost.argmin(dim=2)

    # Bug, if costs over label 0 and 1 are equal, we retain this channel
    # the same as when grad >= threshold (note ==), we retain the channel,
    # also see the notation above find_equal_area()
    channel_label[find_equal_area(cost)] = 1

    mask_size = mask[weight_idx].size()
    if len(mask_size) == 2:
      mask[weight_idx] = channel_label.float()
    else:
      cout, cin, h, w = mask_size
      mask[weight_idx] = channel_label.view(cout, cin, 1, 1).repeat(1, 1, h, w).float()

    if mask[bias_idx] is not None:
      mask[bias_idx] = (channel_label.sum(dim=1) > 0).float()

  mask = resume_dim_conv2fully(mask, grads_org)

  return mask