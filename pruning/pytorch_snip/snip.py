import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types
from prune import *


def SNIP(net, train_dataloader, args):
    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)
    net = net.to(args.device)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    # Zhiwei instead of using random one batch, using the whole dataset to get average grads of mask
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # This is for reproducing, will affect pruning as well as training
            # torch.manual_seed(0)
            # torch.cuda.manual_seed(0)
            # torch.cuda.manual_seed_all(0)

            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))

            if args.weight_init == 'xn':
                nn.init.xavier_normal_(layer.weight)
            elif args.weight_init == 'xu':
                nn.init.xavier_uniform_(layer.weight)
            elif args.weight_init == 'kn':
                nn.init.kaiming_normal_(layer.weight)
            elif args.weight_init == 'ku':
                nn.init.kaiming_uniform_(layer.weight)
            elif args.weight == 'orthogonal':
                nn.init.orthogonal_(layer.weight)
            else:
                assert False

            layer.weight.requires_grad = False

            if layer.bias is not None:
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
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Grab a single batch from the training dataset
    grads_abs_average = []
    for idx, data in enumerate(train_dataloader):
        inputs, targets = data
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        # Compute gradients (but don't apply them)
        net.zero_grad()
        outputs = net.forward(inputs)
        loss = F.nll_loss(outputs, targets)
        loss.backward()


        grads_abs = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grads_abs.append(torch.abs(layer.weight_mask.grad))

                if layer.bias is not None:
                    grads_abs.append(torch.abs(layer.bias_mask.grad))
                else:
                    grads_abs.append(None)

        grads_abs_average = update_grads_average(grads_abs_average, grads_abs, idx)

    return grads_abs_average
