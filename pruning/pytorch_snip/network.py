import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
from torchvision.models import vgg19_bn, resnet152, densenet161


class LeNet_300_100(nn.Module):
  def __init__(self, enable_bias=True):  # original code is true
    super().__init__()
    self.fc1 = nn.Linear(784, 300, bias=enable_bias)
    self.fc2 = nn.Linear(300, 100, bias=enable_bias)
    self.fc3 = nn.Linear(100, 10, bias=enable_bias)

  def forward(self, x):
    x = F.relu(self.fc1(x.view(-1, 784)))
    x = F.relu(self.fc2(x))
    
    return F.log_softmax(self.fc3(x), dim=1)


class LeNet_5(nn.Module):
  def __init__(self, enable_bias=True):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 6, 5, padding=2, bias=enable_bias)
    self.conv2 = nn.Conv2d(6, 16, 5, bias=enable_bias)
    self.fc3 = nn.Linear(16 * 5 * 5, 120, bias=enable_bias)
    self.fc4 = nn.Linear(120, 84, bias=enable_bias)
    self.fc5 = nn.Linear(84, 10, bias=enable_bias)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2)

    x = F.relu(self.fc3(x.view(-1, 16 * 5 * 5)))
    x = F.relu(self.fc4(x))
    x = F.log_softmax(self.fc5(x), dim=1)

    return x

class LeNet_5_Caffe(nn.Module):
  """
  This is based on Caffe's implementation of Lenet-5 and is slightly different
  from the vanilla LeNet-5. Note that the first layer does NOT have padding
  and therefore intermediate shapes do not match the official LeNet-5.
  """

  def __init__(self, enable_bias=True):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 20, 5, padding=0, bias=enable_bias)
    self.conv2 = nn.Conv2d(20, 50, 5, bias=enable_bias)
    self.fc3 = nn.Linear(50 * 4 * 4, 500, bias=enable_bias)
    self.fc4 = nn.Linear(500, 10, bias=enable_bias)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2)

    x = F.relu(self.fc3(x.view(-1, 50 * 4 * 4)))
    x = F.log_softmax(self.fc4(x), dim=1)

    return x


VGG_CONFIGS = {
  'C': [64, 64, 'M', 128, 128, 'M', 256, 256, [256], 'M', 512, 512, [512], 'M', 512, 512, [512], 'M'],
  'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
  'like': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}


class VGG(nn.Module):
  """
  This is a base class to generate three VGG variants used in SNIP paper:
      1. VGG-C (16 layers)
      2. VGG-D (16 layers)
      3. VGG-like

  Some of the differences:
      * Reduced size of FC layers to 512
      * Adjusted flattening to match CIFAR-10 shapes
      * Replaced dropout layers with BatchNorm
  """

  def __init__(self, config, num_classes=10, enable_bias=True, enable_dump_features=False):
    super().__init__()
    self.enable_dump_features = enable_dump_features

    if enable_dump_features:
      self.features_block1 = self.make_layers([64, 64, 'M'], in_channels=3, batch_norm=True, enable_bias=enable_bias)
      self.features_block2 = self.make_layers([128, 128, 'M'], in_channels=64, batch_norm=True, enable_bias=enable_bias)
      self.features_block3 = self.make_layers([256, 256, [256], 'M'], in_channels=128, batch_norm=True, enable_bias=enable_bias)
      self.features_block4 = self.make_layers([512, 512, [512], 'M'], in_channels=256, batch_norm=True, enable_bias=enable_bias)
      self.features_block5 = self.make_layers([512, 512, [512], 'M'], in_channels=512, batch_norm=True, enable_bias=enable_bias)
    else:
      self.features = self.make_layers(VGG_CONFIGS[config], batch_norm=True, enable_bias=enable_bias)

    if config in {'C', 'D'}:
      self.classifier = nn.Sequential(
        nn.Linear(512, 512, bias=enable_bias),  # 512 * 7 * 7 in the original VGG
        nn.ReLU(True),
        nn.BatchNorm1d(512),  # instead of dropout
        nn.Linear(512, 512, bias=enable_bias),
        nn.ReLU(True),
        nn.BatchNorm1d(512),  # instead of dropout
        nn.Linear(512, num_classes, bias=enable_bias))
    elif config == 'like':
      self.classifier = nn.Sequential(
        nn.Linear(512, 512, bias=enable_bias),  # 512 * 7 * 7 in the original VGG
        nn.ReLU(True),
        nn.BatchNorm1d(512),  # instead of dropout
        nn.Linear(512, num_classes, bias=enable_bias))
    else:
      assert False

  @staticmethod
  def make_layers(config, batch_norm=False, enable_bias=True, in_channels=3):  # TODO: BN yes or no?
    layers = []
    for idx, v in enumerate(config):
      if v == 'M':
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
      else:
        if isinstance(v, list):
          v, kernel_size, padding = v[0], 1, 0
        else:
          kernel_size, padding = 3, 1

        conv2d = nn.Conv2d(in_channels, v, kernel_size=kernel_size, padding=padding, bias=enable_bias)
        if batch_norm:
          layers += [conv2d,
                     nn.BatchNorm2d(v),
                     nn.ReLU(inplace=True)]
        else:
          layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)

  def forward(self, input, epoch_id=None, batch_id=None, gt=None):
    if self.enable_dump_features:
      feat_block1 = self.features_block1(input)
      feat_block2 = self.features_block2(feat_block1)
      feat_block3 = self.features_block3(feat_block2)
      feat_block4 = self.features_block4(feat_block3)
      x = self.features_block5(feat_block4)

      if (epoch_id is not None) and (batch_id is not None):
        scio.savemat('../checkpoints/inter_features_epoch{}_batch{}.mat'.format(epoch_id, batch_id),
                     {'img': input.detach().squeeze().permute(2,3,1,0).cpu().numpy(),
                      'gt': gt.detach().squeeze().cpu().numpy(),
                      'b1': feat_block1.detach().squeeze().permute(2,3,1,0).cpu().numpy(),
                      'b2': feat_block2.detach().squeeze().permute(2,3,1,0).cpu().numpy(),
                      'b3': feat_block3.detach().squeeze().permute(2,3,1,0).cpu().numpy(),
                      'b4': feat_block4.detach().squeeze().permute(2,3,1,0).cpu().numpy(),
                      'b5': x.detach().squeeze().cpu().numpy()})
    else:
      x = self.features(input)

    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    x = F.log_softmax(x, dim=1)
    return x


class AlexNet(nn.Module):
  # copy from https://medium.com/@kushajreal/training-alexnet-with-tips-and-checks-on-how-to-train-cnns-practical-cnns-in-pytorch-1-61daa679c74a
  def __init__(self, k=4, num_classes=10, enable_bias=True):
    super(AlexNet, self).__init__()

    self.conv_base = nn.Sequential(
      nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=5, bias=enable_bias),
      nn.BatchNorm2d(96),
      nn.ReLU(inplace=True),

      nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2, bias=enable_bias),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),

      nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=enable_bias),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),

      nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1, bias=enable_bias),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),

      nn.Conv2d(384, 256, kernel_size=3, stride=2, padding=1, bias=enable_bias),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True))

    self.fc_base = nn.Sequential(
      nn.Linear(256, 1024 * k),
      nn.BatchNorm1d(1024 * k),
      nn.ReLU(inplace=True),

      nn.Linear(1024 * k, 1024 * k),
      nn.BatchNorm1d(1024 * k),
      nn.ReLU(inplace=True),

      nn.Linear(1024 * k, num_classes))

  def forward(self, x):
    x = self.conv_base(x)
    x = x.view(x.size(0), -1)
    x = self.fc_base(x)
    x = F.log_softmax(x, dim=1)
    return x