import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

colormap = ['red', 'green', 'yellow', 'blue', 'black', 'blue', 'white']

def viz_voxel(file_name=None, voxel=None, enable_close_time=0, ignore_label=255, mask=None):
  if file_name is not None:
    voxel = scio.loadmat(file_name)['gt']

  classes = np.unique(voxel[voxel != ignore_label])
  colors = np.empty(voxel.shape, dtype=object)

  for idx in range(len(classes)):
    if idx >= len(colormap):
      color = 'white'
    else:
      color = colormap[idx]

    colors[voxel==classes[idx]] = color

  voxel[voxel != ignore_label] = True
  voxel[voxel == ignore_label] = False

  if mask is not None:
    voxel[mask == False] = False

  print('Start to display ...')
  time_start =time.time()
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.voxels(voxel, facecolors=colors, edgecolor='k')
  print('Display time: {:.4f} s'.format(time.time() - time_start))

  if enable_close_time > 0:
    plt.show(block=False)
    plt.pause(enable_close_time)
    plt.close()
  else:
    plt.show()