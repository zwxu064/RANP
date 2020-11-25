import os
import time
import torch
import nibabel as nib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('miccai')

from torch.utils.data import Dataset
from nilearn.image import resample_to_img
from data_augmentation import *
from myutils import RandomCrop3D


def remove_keymap_conflicts(new_keys_set):
  for prop in plt.rcParams:
    if prop.startswith('keymap.'):
      keys = plt.rcParams[prop]
      remove_list = set(keys) & new_keys_set
      for key in remove_list:
        keys.remove(key)


def multi_slice_viewer(mode, volumes, tags):
  remove_keymap_conflicts({'j', 'k'})
  total = volumes[0].shape[0]
  num_volumes = len(volumes)

  if False:  # TODO I do not know why if the first display is empty, it will not update since then
    current_index = volumes[0].shape[0] // 2
  else:
    current_index = 0
    for i in range(volumes[-1].shape[0]):
      if volumes[-1][i].sum() > 0:
        current_index = i
        break

  # rows = int(np.floor(np.sqrt(num_volumes)))
  # cols = int(np.ceil(num_volumes / rows))
  # fig, axes_list = plt.subplots(rows, cols)

  fig, axes_list = plt.subplots(1, num_volumes)

  for i in range(num_volumes):
    volume, tag, axes = volumes[i], tags[i], axes_list[i]
    axes.volume = volume
    axes.index = current_index
    axes.imshow(volume[axes.index])
    axes.title.set_text(tag)

  plt.suptitle('index: {}/{}'.format(current_index, total-1))

  if mode == 'auto':
    fig.canvas.mpl_connect('key_press_event', auto_display)
  else:
    fig.canvas.mpl_connect('key_press_event', process_key)


def auto_display(event):
  fig = event.canvas.figure
  num_subplots = len(fig.axes)
  total = fig.axes[0].volume.shape[0]

  if event.key == 'b':
    for slice in range(total):
      for i in range(num_subplots):
        fig.axes[i].images[0].set_array(fig.axes[i].volume[slice])

      plt.suptitle('index: {}/{}'.format(slice, total-1))
      fig.canvas.draw()
      plt.pause(0.0001)


def process_key(event):
  fig = event.canvas.figure
  num_subplots = len(fig.axes)
  total = fig.axes[0].volume.shape[0]

  if event.key == 'j':
    for i in range(num_subplots):
      previous_slice(fig.axes[i])
  elif event.key == 'k':
    for i in range(num_subplots):
      next_slice(fig.axes[i])

  plt.suptitle('index: {}/{}'.format(fig.axes[0].index, total-1))
  fig.canvas.draw()


def previous_slice(ax):
  volume = ax.volume
  ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
  ax.images[0].set_array(volume[ax.index])


def next_slice(ax):
  volume = ax.volume
  ax.index = (ax.index + 1) % volume.shape[0]
  ax.images[0].set_array(volume[ax.index])


# Label: 1-the necrotic (broken) and non-enhancing tumor core, including 1+4
#        2-the peritumoral edema, surrounding areas including >0
#        4-the GD-enhancing tumor, ==4
class BraTSDataset(Dataset):
  def __init__(self, settype, data_root, years, K=None, enable_time_print=False,
               spatial_size=128, enable_random_crop=False, enable_random_offset=False,
               enable_random_flip=False):
    full_name = 'BraTS'
    self.enable_random_offset = enable_random_offset
    self.enable_random_flip = enable_random_flip
    self.enable_random_crop = enable_random_crop

    if settype in {'test'}:
      self.enable_random_offset = False
      self.enable_random_flip = False
      self.enable_random_crop = False

    for year in years:
      full_name = full_name + '_{}'.format(year)

    list_path = os.path.join(data_root, 'packs', full_name, 'BraTS_{}.txt'.format(settype))
    self.paths = [l.strip('\n') for l in open(list_path).readlines()]
    self.enable_time_print = enable_time_print
    self.settype = settype

    if K is not None:
      self.paths = self.paths[0:K*5]

    if enable_random_crop:
      self.random_crop_3d = RandomCrop3D(spatial_size, enable_crop_random_seed=settype=='train')
    else:
      self.random_crop_3d = None

  def __len__(self):
    return int(len(self.paths) / 5)

  def __getitem__(self, idx):
    if self.enable_time_print:
      time_start = time.time()

    name = self.paths[5 * idx].split('/')[-1]

    # Read data
    obj = nib.load(self.paths[5 * idx])
    t1 = obj.get_fdata()

    obj = nib.load(self.paths[5 * idx + 1])
    t2 = obj.get_fdata()

    obj = nib.load(self.paths[5 * idx + 2])
    t1ce = obj.get_fdata()

    obj = nib.load(self.paths[5 * idx + 3])
    flair = obj.get_fdata()

    obj = nib.load(self.paths[5 * idx + 4])
    label = obj.get_fdata()  # TODO

    affine = obj.affine

    data = np.stack((t1, t2, t1ce, flair), axis=0)
    label = np.expand_dims(label, 0)

    # Augmentation
    n_dim = len(label[0].shape)
    offset_factor = -0.65 + np.random.random(n_dim) if self.enable_random_offset else None
    flip_axis = random_flip_dimensions(n_dim) if self.enable_random_flip else None
    data_list = []

    for data_channel in range(data.shape[0]):
      # Transform ndarray data to Nifti1Image
      channel_image = nib.Nifti1Image(dataobj=data[data_channel], affine=affine)
      data_list.append(resample_to_img(augment_image(channel_image, flip_axis=flip_axis, offset_factor=offset_factor), channel_image, interpolation="continuous").get_data())

    data = np.asarray(data_list)

    # Transform ndarray segmentation label to Nifti1Image
    seg_image = nib.Nifti1Image(dataobj=label[0], affine=affine)
    label = resample_to_img(augment_image(seg_image, flip_axis=flip_axis, offset_factor=offset_factor), seg_image, interpolation="nearest").get_data()
    label = np.expand_dims(label, 0) if (len(label.shape) == 3) else label

    if self.settype == "test":
      print('Test set TBD')

    # RandomCrop3D
    if self.random_crop_3d is not None:
      data, label = self.random_crop_3d([data, label])

    data = torch.from_numpy(data).float()
    label = torch.from_numpy(label).long()

    # Normalization to [-1, 1]
    data_max, _ = data.view(data.size(0), -1).max(1)
    data_max = data_max.view(data.size(0), 1, 1, 1)
    data = data / (data_max + 1e-8)
    data = (data - 0.5) * 2

    sample = {'x': data, 'y': label, 'affine': affine, 'name': name}

    if self.enable_time_print:
      print('time all: {}'.format(time.time() - time_start))

    return sample


if __name__ == '__main__':
  # # Train
  # dataset = BraTSDataset('train', '../../datasets/BraTS', [2019], spatial_size=128,
  #                        enable_random_offset=False,
  #                        enable_random_crop=True,
  #                        enable_random_flip=True)
  # obj = dataset.__getitem__(0)
  # data, gt = obj['x'], obj['y']
  # print(obj['name'], data.shape, gt.shape, data.min(), data.max())
  # data = np.transpose(data, [0, 3, 2, 1])
  # gt = np.transpose(gt, [0, 3, 2, 1])
  #
  # multi_slice_viewer('auto', [data[0], data[1], data[2], data[3], gt[0]], ['t1', 't1ce', 't2', 'flair', 'seg'])
  # plt.show()

  # Validation
  validset = BraTSDataset('valid', '../../datasets/BraTS', [2018, 2019], spatial_size=128,
                          enable_random_offset=False,
                          enable_random_crop=True,
                          enable_random_flip=False)

  obj = validset.__getitem__(1)
  data, gt = obj['x'], obj['y']
  print(obj['name'], data.shape, gt.shape, data.min(), data.max())
  data = np.transpose(data, [0, 3, 2, 1])
  gt = np.transpose(gt, [0, 3, 2, 1])

  multi_slice_viewer('auto', [data[0], data[1], data[2], data[3], gt[0]], ['t1', 't1ce', 't2', 'flair', 'seg'])
  plt.show()
