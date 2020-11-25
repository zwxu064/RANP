from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms


def get_mnist_dataloaders(train_batch_size, val_batch_size, args, enable_train_shuffle=True):
  data_transform = Compose([transforms.ToTensor()])

  # Normalise? transforms.Normalize((0.1307,), (0.3081,))

  train_dataset = MNIST("{}/{}".format(args.relative_dir, "../dataset"), True, data_transform, download=True)
  test_dataset = MNIST("{}/{}".format(args.relative_dir, "../dataset"), False, data_transform, download=False)

  train_loader = DataLoader(train_dataset, train_batch_size, shuffle=enable_train_shuffle, num_workers=2, pin_memory=True)
  test_loader = DataLoader(test_dataset, val_batch_size, shuffle=False, num_workers=2, pin_memory=True)

  return train_loader, test_loader


def get_cifar10_dataloaders(train_batch_size, test_batch_size, args, enable_train_shuffle=True, enable_train_trans=True):
  if enable_train_trans:
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  else:
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

  test_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

  train_dataset = CIFAR10("{}/{}".format(args.relative_dir, "../dataset"), True, train_transform, download=True)
  test_dataset = CIFAR10("{}/{}".format(args.relative_dir, "../dataset"), False, test_transform, download=False)

  train_loader = DataLoader(train_dataset, train_batch_size, shuffle=enable_train_shuffle, num_workers=2, pin_memory=True)
  test_loader = DataLoader(test_dataset, test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

  return train_loader, test_loader