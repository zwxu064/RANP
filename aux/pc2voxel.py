import numpy as np
import numpy.random as np_random
import time
from aux.data import load, classOffsets, nClasses
from aux.binvox_rw import Voxels
from aux.viz_voxel import viz_voxel


def random_trans(left, right):
  return np_random.random() * (right - left) + left

def point_2_voxel(spatial_size, dim, points, scale, ignore_label=255, result_dir=None,
                  trans=None, enable_random_trans=False, enable_random_rotate=False,
                  gt=None, enable_debug=False, enable_voxel_gt=False,
                  enable_hard_padding=False):
  assert dim * scale <= spatial_size
  assert points.min() >= -1 and points.max() <= 1
  enable_voxel_gt = None if (gt is None) else enable_voxel_gt
  trans = np.array([[0.,0.,0.]], dtype=np.float64).T if (trans is None) else trans
  points = points.T if (points.shape[0] != 3) else points
  gt = gt.T if (gt.shape[0] != 1) else gt
  trans = trans.T if (not enable_random_trans) and (trans.shape[0] != 3) else trans
  n_points = points.shape[1]

  # Random rotate, move before translate to avoid index overflow
  if enable_random_rotate:
    m = np.eye(3, dtype='float32')
    m[0, 0] *= np_random.randint(0, 2) * 2 - 1
    m = np.dot(m, np.linalg.qr(np_random.randn(3, 3))[0])
    points = np.dot(m, points)

  # Scale > 1 is too slow
  if enable_hard_padding:
    assert scale > 1, 'Error! For hard padding, scale>1 while scale:{}'.format(scale)
    assert dim <= spatial_size,\
      'Error! For hard padding, dim<=spatial size while dim:{}, spatial size:{}'.format(dim, spatial_size)
    spatial_size_org = spatial_size
    spatial_size = dim
    scale = 1

  # points (3,n_points), formula is (points + trans) * (dim / 2) + spatial_size / 2 in [0, spatial_size]
  tmp = (spatial_size / 2) / (dim / 2)
  max_trans_p_left = -tmp - points.min(1)
  max_trans_p_right = tmp - points.max(1)

  if enable_debug:
    print(max_trans_p_left, max_trans_p_right)

  if enable_random_trans:
    trans = np.array([[random_trans(max_trans_p_left[i], max_trans_p_right[i])] for i in range(3)], dtype=trans.dtype)
  elif enable_debug:
    assert (trans[0] >= max_trans_p_left[0]) and (trans[0] <= max_trans_p_right[0]), 'x range in [{:.4f},{:.4f}]'.format(max_trans_p_left[0], max_trans_p_right[0])
    assert (trans[1] >= max_trans_p_left[1]) and (trans[1] <= max_trans_p_right[1]), 'y range in [{:.4f},{:.4f}]'.format(max_trans_p_left[1], max_trans_p_right[1])
    assert (trans[2] >= max_trans_p_left[2]) and (trans[2] <= max_trans_p_right[2]), 'z range in [{:.4f},{:.4f}]'.format(max_trans_p_left[2], max_trans_p_right[2])

  # Central moving is contained in +spatial_size // 2
  points = np.floor((points + trans) * (dim / 2) + spatial_size / 2)
  expand_list = np.arange(0, scale) - scale // 2  # keep side equal distances
  expand_list = np.repeat(np.expand_dims(expand_list, 0), 3, axis=0)
  expand_points = []
  expand_gt = []

  if (scale > 1) or (gt is not None):
    for idx in range(n_points):
      expand_one = points[:,idx:idx+1] + expand_list
      x_list = np.unique(expand_one[0, :])
      y_list = np.unique(expand_one[1, :])
      z_list = np.unique(expand_one[2, :])
      expand_one = np.array(np.meshgrid(x_list, y_list, z_list)).reshape(3, -1)
      expand_points.append(expand_one)

      if gt is not None:
        gt_cat = np.vstack([np.concatenate([expand_one[:,num], gt[:,idx]]) for num in range(expand_one.shape[1])])
        expand_gt.append(gt_cat) if gt is not None else None

    if gt is not None:
      expand_gt = np.vstack(expand_gt).T

    expand_points = np.hstack(expand_points)
  else:
    expand_points = points

  coord, indice, counts = np.unique(expand_points, axis=1, return_index=True, return_counts=True)
  index = coord[0,:]*(spatial_size**2) + coord[1,:]*spatial_size + coord[2,:]
  voxel = np.zeros((spatial_size, spatial_size, spatial_size), dtype=bool)
  voxel = voxel.flatten()
  voxel[np.int32(index)] = True
  voxel = voxel.reshape(spatial_size, spatial_size, spatial_size)

  mode = 'xzy'
  voxel = np.transpose(voxel, (0,2,1)) if mode == 'xzy' else voxel

  if gt is not None:
    # Assign label from voxels taken up the most, when two voxels with different labels merged to one voxel
    overlaps = coord[:, counts > 1]  # 23-Jan-2020, large number of overlaps make it very slow
    for over_idx in range(overlaps.shape[1]):
      overlap = overlaps[:, over_idx].reshape(-1,1)
      area = np.all(expand_gt[0:3, :] == overlap, axis=0)
      value = expand_gt[:, area]
      uni_data, uni_counts = np.unique(value, axis=1, return_counts=True)
      if len(uni_counts) == 1: continue
      max_index = np.argmax(uni_counts)
      expand_gt[:, area] = uni_data[:, max_index].reshape(-1,1)

    if enable_voxel_gt:
      voxel_gt = 255 * np.ones((spatial_size, spatial_size, spatial_size), dtype=bool)
      voxel_gt = voxel_gt.flatten()
      voxel_gt[np.int32(index)] = expand_gt[3, indice]
      voxel_gt = voxel_gt.reshape(spatial_size, spatial_size, spatial_size)
      voxel_gt = np.transpose(voxel_gt, (0,2,1)) if mode == 'xzy' else voxel_gt
      assert (voxel == True).sum() == (voxel_gt != ignore_label).sum()
      # file_path = '{}/test_dim{}_size{}_scale{}.mat'.format(result_dir, dim, spatial_size, scale)
      # scio.savemat(file_path, {'gt': expand_gt[3,:]})
    else:
      voxel_gt = None

  if result_dir is not None:
    object = Voxels(voxel, [str(spatial_size), str(spatial_size), str(spatial_size)], [str(0), str(0), str(0)], str(spatial_size), mode)
    print('spatial size', object.data.shape)
    file_path = '{}/test_dim{}_size{}_scale{}.binvox'.format(result_dir, dim, spatial_size, scale)

    with open(file_path, 'w') as fp:
      object.write(fp)

  if enable_hard_padding:
    voxel_padded = np.zeros((spatial_size_org, spatial_size_org, spatial_size_org), dtype=bool)
    offset_start = dim // 2
    offset_end = offset_start + dim
    voxel_padded[offset_start:offset_end, offset_start:offset_end, offset_start:offset_end] = voxel
    del voxel
    voxel = voxel_padded

    if voxel_gt is not None:
      gt_padded = 255 * np.ones((spatial_size_org, spatial_size_org, spatial_size_org), dtype=bool)
      gt_padded[offset_start:offset_end, offset_start:offset_end, offset_start:offset_end] = voxel_gt
      del voxel_gt
      voxel_gt = gt_padded

  if enable_voxel_gt:
    return voxel, voxel_gt
  else:
    expand_gt_Np = 255 * np.ones((1, 3000), dtype='int64')
    expand_gt_Np[0, :n_points] = expand_gt[-1, :].reshape(1, -1)
    return voxel, expand_gt_Np


if __name__ == '__main__':
  target_class = 0
  file_name = 'train_val/02691156/000001.pts.train'
  result_dir = '/home/users/u5710355/WorkSpace/git-lab/matlab-projects/binvox-rw-matlab'
  norm_points = load([file_name], target_class, classOffsets[target_class], nClasses[target_class])
  input, gt = norm_points[1:3]
  gt = np.array([gt])

  # Small case
  # dim, spatial_size, scale = 4, 8, 1
  # input = np.zeros((3,2), dtype=np.float64)
  # input[:,0] = [-0.5,0,0]
  # input[:,1] = [0.5,0,0]
  # voxel, voxel_gt = point_2_voxel(spatial_size, dim, input, scale, result_dir)
  # viz_voxel(voxel=voxel)

  # Case 1: from point cloud XYZ file
  dim, spatial_size, scale = 32, 32, 1
  trans = np.array([[0.,0.,0.]], dtype=np.float64).T
  voxel, voxel_gt = point_2_voxel(spatial_size, dim, input, scale, result_dir, trans=trans, gt=gt,
                                  enable_random_trans=False, enable_random_rotate=False,
                                  enable_voxel_gt=True)
  viz_voxel(voxel=voxel_gt)
