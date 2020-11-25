import os, glob, torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from aux.data import categories, classOffsets, nClasses
from aux.pc2voxel import point_2_voxel
from aux.viz_voxel import viz_voxel


def read_pcf(xF, class_offset, mode):
  input = np.loadtxt(xF)
  input /= ((input ** 2).sum(1).max() ** 0.5)

  if mode in ['train', 'valid']:
    filename_format = 'pts.{}'.format(mode)
  else:
    filename_format = 'pts'

  gt = np.loadtxt(xF.replace(filename_format, 'seg')).astype('int64') + class_offset - 1
  gt = gt.reshape(-1, 1)
  return input, gt


def get_datalist(mode, target_class, data_dir):
  list = []
  if mode in ['train', 'valid']:
    data_dir = '{}/train_val'.format(data_dir)
    filename = '*.pts.{}'.format(mode)
  else:
    data_dir = '{}/test'.format(data_dir)
    filename = '*.pts'

  if target_class < 0:
    for c in range(16):
      paths = glob.glob(
        os.path.join(data_dir, categories[c], filename))
      for path in paths:
        list.append({'path': path, 'class': c})
  else:
    paths = glob.glob(os.path.join(data_dir, categories[target_class], filename))
    for path in paths:
      list.append({'path': path, 'class': target_class})

  return list


class SHAPENET(Dataset):
  def __init__(self, mode, spatial_size, dim, scale, data_dir, target_class=-1,
               enable_random_trans=False, enable_random_rotate=False,
               enable_voxel_gt=False, enable_hard_padding=False):
    self.mode = mode
    self.spatial_size = spatial_size
    self.dim = dim
    self.scale = scale
    self.enable_random_trans = enable_random_trans
    self.enable_random_rotate = enable_random_rotate
    self.enable_voxel_gt = enable_voxel_gt
    self.datalist = get_datalist(mode, target_class, data_dir)
    self.enable_hard_padding = enable_hard_padding

  def __getitem__(self, index):
    data = self.datalist[index]
    file_path, target_class = data['path'], data['class']
    points, gt = read_pcf(file_path, classOffsets[target_class], mode=self.mode)
    voxels, gt = point_2_voxel(self.spatial_size, self.dim, points, self.scale,
                               gt=gt, result_dir=None, trans=None, ignore_label=255,
                               enable_random_trans=self.enable_random_trans,
                               enable_random_rotate=self.enable_random_rotate,
                               enable_voxel_gt=self.enable_voxel_gt, enable_debug=False,
                               enable_hard_padding=self.enable_hard_padding)

    # Do not create new one, just use SSC
    voxels = np.expand_dims(voxels.astype(np.float32), axis=0)  # create channel as 1
    voxels = torch.from_numpy(voxels)
    gt = torch.from_numpy(gt).long()

    return {'x': voxels, 'y': gt, 'file_path': file_path, 'num_class': nClasses[target_class],
            'class_offset': classOffsets[target_class]}

  def __len__(self):
    return len(self.datalist)

if __name__ == '__main__':
  import time
  spatial_size, dim, scale = 128, 128, 1
  mode, data_dir, batch_size = 'train', 'datasets/ShapeNet', 1
  dataset = SHAPENET(mode, spatial_size, dim, scale, data_dir, target_class=6,
                     enable_random_trans=False, enable_random_rotate=False, enable_voxel_gt=True,
                     enable_hard_padding=True)
  dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=False)
  enable_time = True

  time_start = time.time()
  count = 1
  for idx, data in enumerate(dataloader):
    if idx != 0: continue
    if not enable_time:
      print(idx, data['file_path'])
      voxel_input, gt = data['x'], data['y']
      voxel_input[voxel_input==0] = 255  # to display

      if (len(gt.shape) == 4) and (batch_size == 1):
        voxel_gt = gt.reshape(spatial_size, spatial_size, spatial_size).cpu().numpy()
        viz_voxel(voxel=voxel_gt, enable_close_time=-1)
      else:
        print(gt.shape)
    else:
      count += 1
      voxel_input, gt = data['x'], data['y']

  if enable_time: print('Total time: {:.4f}s for {} samples'.format((time.time() - time_start) / count, count))
