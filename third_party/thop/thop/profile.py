import logging

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from .count_hooks import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,

    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,

    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,

    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,

    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,

    nn.Linear: count_linear,
    nn.Dropout: zero_ops,

    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample,

    nn.Softmax: count_softmax
}


def register_count_memory(self, input, output):
    self.total_memory += output.nelement()


def profile(model,
            inputs,
            custom_ops=None,
            verbose=True,
            enable_layer_neuron_display=False,
            resource_list_type=None):
    handler_collection = []
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        if hasattr(m, "total_ops") \
                or hasattr(m, "total_params") \
                or hasattr(m, "total_memory") \
                or hasattr(m, "output_size"):
            logger.warning("Either .total_ops or .total_params is already defined in %s."
                           "Be careful, it might change your code's behavior." % str(m))

        m.register_buffer('total_ops', torch.zeros(1, dtype=torch.double))
        m.register_buffer('total_params', torch.zeros(1, dtype=torch.double))
        m.register_buffer('total_memory', torch.zeros(1, dtype=torch.double))
        m.register_buffer('output_size', None)

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()]).double()

        m_type = type(m)
        fn = None
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]

        if fn is None:
            if verbose:
                print("THOP has not implemented counting method for ", m)
        else:
            if verbose:
                print("Register FLOP counter for module %s" % str(m))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

        # m.register_forward_hook(register_count_memory)
        handler = m.register_forward_hook(register_count_memory)
        handler_collection.append(handler)

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    total_ops, total_params, total_memory = 0, 0, 0
    layer_id = 0
    enable_resource_list = any([resource_list_type.find(s) > -1 for s in ['flops', 'param', 'memory']]) \
      if (resource_list_type is not None) else False
    resource_list = [] if enable_resource_list else None

    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        opt_layer = m.total_ops.double()
        param_layer = m.total_params.double()
        memory_layer = m.total_memory.double()
        output_size_layer = m.output_size

        total_ops += opt_layer
        total_params += param_layer
        total_memory += memory_layer

        if enable_resource_list and isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d,
                                                   nn.ConvTranspose3d, nn.ConvTranspose1d, nn.Linear)):
          if resource_list_type.find('flops') > -1:
            resource_layer = opt_layer
          elif resource_list_type.find('param') > -1:
            resource_layer = param_layer
          elif resource_list_type.find('memory') > -1:
            resource_layer = memory_layer
          else:
            resource_layer = None

          resource_list.append(resource_layer)

        if enable_layer_neuron_display:
            neuron = 0
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                m_type = 'Conv'
                neuron = m.out_channels
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m_type = 'BNorm'
            elif isinstance(m, (nn.ReLU, nn.ReLU6)):
                m_type = 'Activ'
            elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
                m_type = 'MaxPool'
            elif isinstance(m, (nn.Softmax)):
                m_type = 'Softmax'
            else:
                m_type = 'Unknown'

            print('Profile, layer id: {}, type: {}, neuron: {}, FLOPS: {:.2f}G, param: {:.2f}KB, memory: {:.2f}MB, output size: {}'. \
                  format(layer_id,
                         m_type,
                         neuron,
                         opt_layer.item() / 1e9,
                         param_layer.item() * 4 / 1024,
                         memory_layer.item() * 4 / (1024 ** 2),
                         output_size_layer.numpy()))
        layer_id += 1

    resource_list = torch.cat(resource_list, 0) if (resource_list is not None) else None
    total_ops = total_ops.item()
    total_params = total_params.item()
    total_memory = total_memory.item()

    # reset model to original status
    model.train(training)
    for handler in handler_collection:
        handler.remove()

    # remove temporal buffers
    for n, m in model.named_modules():
        if len(list(m.children())) > 0:
            continue
        if "total_ops" in m._buffers:
            m._buffers.pop("total_ops")
        if "total_params" in m._buffers:
            m._buffers.pop("total_params")
        if "total_memory" in m._buffers:
            m._buffers.pop("total_memory")
        if "output_size" in m._buffers:
            m._buffers.pop("output_size")

    return total_ops, total_params, total_memory, resource_list
