import os
import pickle
import shutil


def check_dir(dir_name):
  if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def create_test_pkl(data_root, dataset, years):
  return 0


def create_train_valid_pkl(data_root, dataset, years, split_ratio=None):
  assert os.path.exists(data_root)
  dataset_dir = os.path.join(data_root, dataset)
  assert os.path.exists(dataset_dir)
  full_name = "BraTS"

  for year in years:
    full_name = full_name + '_{}'.format(year)

  path_file_dir = os.path.join(dataset_dir, 'packs')
  check_dir(path_file_dir)
  path_file_dir = os.path.join(path_file_dir, full_name)
  check_dir(path_file_dir)

  print('====Creating train+valid pkl')
  paths = []

  for year in years:
    data_dir = os.path.join(dataset_dir, '{}'.format(year))
    data_dir = os.path.join(data_dir, 'MICCAI_BraTS_{}_Data_Training'.format(year))

    for root, dirs, files in os.walk(data_dir):
      for file in files:
        if file.find('.nii.gz') > -1:
          id_name = root.split('/')[-1]
          paths.append(os.path.join(root, '{}_t1.nii.gz'.format(id_name)))
          paths.append(os.path.join(root, '{}_t1ce.nii.gz'.format(id_name)))
          paths.append(os.path.join(root, '{}_t2.nii.gz'.format(id_name)))
          paths.append(os.path.join(root, '{}_flair.nii.gz'.format(id_name)))
          paths.append(os.path.join(root, '{}_seg.nii.gz'.format(id_name)))
          break

  # 1 in split_ratio percent for valid, the rest for train
  train_paths, valid_paths = [], []
  for path_ind in range(int(len(paths) / 5)):
    if (split_ratio is not None) and (path_ind % split_ratio == 0):
      for i in range(5):
        valid_paths.append(paths[5 * path_ind + i])
    else:
      for i in range(5):
        train_paths.append(paths[5 * path_ind + i])

  fout = open(os.path.join(path_file_dir, '{}_train.pkl'.format(dataset)), 'wb')
  pickle.dump(train_paths, fout)
  fout.close()

  fout = open(os.path.join(path_file_dir, '{}_valid.pkl'.format(dataset)), 'wb')
  pickle.dump(valid_paths, fout)
  fout.close()

  print('Dataset: {}, train images: {}, valid images: {}'.
        format(dataset, int(len(train_paths) / 5), int(len(valid_paths) / 5)))

  # Save file paths to txt
  fp = open(os.path.join(path_file_dir, '{}_train.txt'.format(dataset)), 'w')
  for path_ind in range(len(train_paths)):
    fp.write(str(train_paths[path_ind]))
    fp.write('\n')
  fp.close()

  fp = open(os.path.join(path_file_dir, '{}_valid.txt'.format(dataset)), 'w')
  for path_ind in range(len(valid_paths)):
    fp.write(str(valid_paths[path_ind]))
    fp.write('\n')
  fp.close()

  if dataset.find('-single') > -1:
    return

  # Check
  print('====Checking')
  fin = open(os.path.join(path_file_dir, '{}_train.pkl'.format(dataset)), 'rb')
  train_paths = pickle.load(fin)
  fin.close()

  fin = open(os.path.join(path_file_dir, '{}_valid.pkl'.format(dataset)), 'rb')
  valid_paths = pickle.load(fin)
  fin.close()

  all_paths = train_paths + valid_paths
  path_len = len(all_paths)
  assert (path_len % 5 == 0)

  for path in all_paths:
    assert os.path.exists(path)


if __name__ == '__main__':
  dataset = 'BraTS'
  server = '039614'
  data_root = []

  if server == 'data61':
    data_root = '/flush3/xu064/zhiwei/Datasets'
  elif server == '039614':
    data_root = '/mnt/scratch/zhiwei/Datasets'

  create_train_valid_pkl(data_root, dataset, [2019], split_ratio=5)
  create_test_pkl(data_root, dataset, [2019])
