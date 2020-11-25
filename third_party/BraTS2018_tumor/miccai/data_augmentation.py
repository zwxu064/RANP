'''
@Author: Zhou Kai
@GitHub: https://github.com/athon2
@Date: 2018-11-30 09:53:44
'''
from torch.utils.data import Dataset
import tables
import numpy as np
import nibabel as nib
from nilearn.image import new_img_like, resample_to_img


def random_flip_dimensions(n_dimensions):
  axis = list()
  for dim in range(n_dimensions):
    if np.random.choice([True, False]):
      axis.append(dim)

  return axis


def flip_image(image, axis):
  try:
    new_data = np.copy(image.get_data())
    for axis_index in axis:
      new_data = np.flip(new_data, axis=axis_index)
  except TypeError:
    new_data = np.flip(image.get_data(), axis=axis)

  return new_img_like(image, data=new_data)


def offset_image(image, offset_factor):
  image_data = image.get_data()
  image_shape = image_data.shape
  new_data = np.zeros(image_shape)

  assert len(image_shape) == 3, "Wrong dimessions! Expected 3 but got {0}".format(len(image_shape))

  if len(image_shape) == 3:
    new_data[:] = image_data[0][0][0]
    oz = int(image_shape[0] * offset_factor[0])
    oy = int(image_shape[1] * offset_factor[1])
    ox = int(image_shape[2] * offset_factor[2])

    if oy >= 0:
      slice_y = slice(image_shape[1]-oy)
      index_y = slice(oy, image_shape[1])
    else:
      slice_y = slice(-oy,image_shape[1])
      index_y = slice(image_shape[1] + oy)

    if ox >= 0:
      slice_x = slice(image_shape[2]-ox)
      index_x = slice(ox, image_shape[2])
    else:
      slice_x = slice(-ox,image_shape[2])
      index_x = slice(image_shape[2] + ox)

    if oz >= 0:
      slice_z = slice(image_shape[0]-oz)
      index_z = slice(oz, image_shape[0])
    else:
      slice_z = slice(-oz,image_shape[0])
      index_z = slice(image_shape[0] + oz)

    new_data[index_z, index_y, index_x] = image_data[slice_z, slice_y, slice_x]

  return new_img_like(image, data=new_data)


def augment_image(image, flip_axis=None, offset_factor=None):
  if flip_axis is not None:
    image = flip_image(image, axis=flip_axis)

  if offset_factor is not None:
    image = offset_image(image, offset_factor=offset_factor)

  return image


def get_target_label(label_data, config):
  target_label = np.zeros(label_data.shape)

  for l_idx in range(config["n_labels"]):
    assert config["labels"][l_idx] in [1, 2, 4], \
      "Wrong label! Expected 1 or 2 or 4, but got {0}".format(config["labels"][l_idx])

    if not config["label_containing"]:
      target_label[np.where(label_data == config["labels"][l_idx])] = 1
    else:
      if config["labels"][l_idx] == 1:
        target_label[np.where(label_data == 1)] = 1
        target_label[np.where(label_data == 4)] = 1
      elif config["labels"][l_idx] == 2:
        target_label[np.where(label_data > 0)] = 1
      elif config["labels"][l_idx] == 4:
        target_label[np.where(label_data == 4)] = 1

  return target_label
