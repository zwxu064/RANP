# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch, data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import os, sys
import math
import numpy as np


class Model(nn.Module):
  def __init__(self):
    nn.Module.__init__(self)
    self.input = scn.Sequential().add(
      scn.InputLayer(dimension, data.spatialSize, mode=3))

    self.sparseModel = scn.Sequential().add(
      scn.SubmanifoldConvolution(dimension, 1, m, 3, False)).add(
      scn.FullyConvolutionalNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[3,2])).add(
      scn.BatchNormReLU(sum(nPlanes))).add(
      scn.OutputLayer(dimension))

    # self.sparseModel = scn.Sequential().add(
    #   scn.InputLayer(dimension, data.spatialSize, mode=3)).add(
    #   scn.SubmanifoldConvolution(dimension, 1, m, 3, False)).add(
    #   scn.FullyConvolutionalNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[3,2])).add(
    #   scn.BatchNormReLU(sum(nPlanes))).add(
    #   scn.OutputLayer(dimension))
    self.linear = nn.Linear(sum(nPlanes), data.nClassesTotal)
  def forward(self,x):
    x = self.input(x)
    x = self.sparseModel(x)
    # x=self.sparseModel(x)
    x=self.linear(x)
    return x


def store(stats,batch,predictions,loss):
  ctr=0
  for nP,f,classOffset,nClasses in zip(batch['nPoints'],batch['xf'],batch['classOffset'],batch['nClasses']):
    categ,f=f.split('/')[-2:]
    if not categ in stats:
      stats[categ]={}
    if not f in stats[categ]:
      stats[categ][f]={'p': 0, 'y': 0}
    #print(predictions[ctr:ctr+nP,classOffset:classOffset+nClasses].abs().max().item())
    stats[categ][f]['p']+=predictions.detach()[ctr:ctr+nP,classOffset:classOffset+nClasses].cpu().numpy()
    stats[categ][f]['y']=batch['y'].detach()[ctr:ctr+nP].cpu().numpy()-classOffset
    ctr+=nP


# Zhiwei
def store_voxel(stats, data, predictions, gts):
  # Predictions (batch,class,dim^3), class first
  predictions = predictions.detach().cpu().numpy()
  gts = gts.detach().cpu().numpy()
  batch = predictions.shape[0]
  actual_batch = len(data['class_offset'])

  for idx in range(batch):
    class_offset, num_class = data['class_offset'][idx].cpu().numpy(), data['num_class'][idx].cpu().numpy()
    prediction, gt = predictions[idx], gts[idx]
    categ, f = data['file_path'][idx].split('/')[-2:]

    if not categ in stats:
      stats[categ]={}

    if not f in stats[categ]:
      stats[categ][f]={'p': 0, 'y': 0}

    mask = (gt != 255)  # ignore empty space
    stats[categ][f]['p'] += prediction[class_offset:class_offset+num_class, mask]
    stats[categ][f]['y'] = gt[mask] - class_offset


def inter(pred, gt, label):
  assert pred.size == gt.size, 'Predictions incomplete!'
  return np.sum(np.logical_and(pred.astype('int') == label, gt.astype('int') == label))


def union(pred, gt, label):
  assert pred.size == gt.size, 'Predictions incomplete!'
  return np.sum(np.logical_or(pred.astype('int') == label, gt.astype('int') == label))


def iou(stats, class_first=False):
  eps = sys.float_info.epsilon
  categories= sorted(stats.keys())
  ncategory = len(categories)
  iou_all = np.zeros(ncategory)
  nmodels = np.zeros(ncategory, dtype='int')
  for i, categ in enumerate(categories):
    nmodels[i] = len(stats[categ])
    pred = []
    gt = []
    for j in stats[categ].values():
      if class_first:
        pred.append(j['p'].argmax(0))
      else:
        pred.append(j['p'].argmax(1))
      gt.append(j['y'])
    npart = np.max(np.concatenate(gt))+1
    iou_per_part = np.zeros((len(pred), npart))
    # loop over parts
    for j in range(npart):
      # loop over CAD models
      for k in range(len(pred)):
        p = pred[k]
        iou_per_part[k, j] = (inter(p, gt[k], j) + eps) / (union(p, gt[k], j) + eps)
    # average over CAD models and parts
    iou_all[i] = np.mean(iou_per_part)
  # weighted average over categories
  iou_weighted_ave = np.sum(iou_all * nmodels) / np.sum(nmodels)
  return {'iou': iou_weighted_ave, 'nmodels_sum': nmodels.sum(), 'iou_all': iou_all}


if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"] = "2"
  torch.manual_seed(2019)
  torch.cuda.manual_seed_all(2019)
  np.random.seed(2019)
  enable_cuda = False  # torch.cuda.is_available()
  device = torch.device('cuda' if enable_cuda else 'cpu')
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  import sparseconvnet as scn

  checkpoint_dir = 'fcn_ssc_checkpoint'
  if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

  resolution = 24
  spartial_size = resolution * 8 + 15
  print('Resolution:{}, spartial size:{}'.format(resolution, spartial_size))
  data.init(-1, resolution, spartial_size, 16)
  dimension = 3
  reps = 2  # Conv block repetition factor
  m = 32  # Unet number of features
  nPlanes = [m, 2 * m, 3 * m, 4 * m, 5 * m]  # UNet number of features per level

  model = Model()
  # print(model)
  trainIterator = data.train()
  validIterator = data.valid()

  criterion = nn.CrossEntropyLoss()
  p = {}
  p['n_epochs'] = 100
  p['initial_lr'] = 1e-1
  p['lr_decay'] = 4e-2
  p['weight_decay'] = 1e-4
  p['momentum'] = 0.9
  p['check_point'] = True
  p['use_cuda'] = False  # torch.cuda.is_available()
  dtype = 'torch.cuda.FloatTensor' if p['use_cuda'] else 'torch.FloatTensor'
  dtypei = 'torch.cuda.LongTensor' if p['use_cuda'] else 'torch.LongTensor'
  if p['use_cuda']:
    model.cuda()
    criterion.cuda()
  optimizer = optim.SGD(model.parameters(),
                        lr=p['initial_lr'],
                        momentum=p['momentum'],
                        weight_decay=p['weight_decay'],
                        nesterov=True)
  if p['check_point'] and os.path.isfile('epoch.pth'):
    p['epoch'] = torch.load('epoch.pth') + 1
    print('Restarting at epoch ' +
          str(p['epoch']) +
          ' from model.pth ..')
    model.load_state_dict(torch.load('model.pth'))
  else:
    p['epoch'] = 1
  print(p)
  print('#parameters', sum([x.nelement() for x in model.parameters()]))

  for epoch in range(p['epoch'], p['n_epochs'] + 1):
    model.train()
    stats = {}
    for param_group in optimizer.param_groups:
      param_group['lr'] = p['initial_lr'] * \
                          math.exp((1 - epoch) * p['lr_decay'])
    scn.forward_pass_multiplyAdd_count=0
    scn.forward_pass_hidden_states=0
    start = time.time()
    for batch in trainIterator:
      optimizer.zero_grad()
      batch['x'][1]=batch['x'][1].type(dtype)
      batch['y']=batch['y'].type(dtypei)
      batch['mask']=batch['mask'].type(dtype)
      predictions=model(batch['x'])
      loss = criterion.forward(predictions,batch['y'])
      store(stats,batch,predictions,loss)
      loss.backward()
      optimizer.step()
    r = iou(stats)
    print('train epoch',epoch,1,'iou=', r['iou'], 'MegaMulAdd=',scn.forward_pass_multiplyAdd_count/r['nmodels_sum']/1e6, 'MegaHidden',scn.forward_pass_hidden_states/r['nmodels_sum']/1e6,'time=',time.time() - start,'s')

    if p['check_point']:
      # torch.save(epoch, 'epoch.pth')
      # torch.save(model.state_dict(),'model.pth')
      torch.save(model.state_dict(),
                 os.path.join(checkpoint_dir, 'model_epoch{}.pth'.format(epoch)))

    if epoch in [10,30,100]:
      model.eval()
      stats = {}
      scn.forward_pass_multiplyAdd_count=0
      scn.forward_pass_hidden_states=0
      start = time.time()
      for rep in range(1,1+3):
        for batch in validIterator:
          batch['x'][1]=batch['x'][1].type(dtype)
          batch['y']=batch['y'].type(dtypei)
          batch['mask']=batch['mask'].type(dtype)
          predictions=model(batch['x'])
          loss = criterion.forward(predictions,batch['y'])
          store(stats,batch,predictions,loss)
        r = iou(stats)
        print('valid epoch',epoch,rep,'iou=', r['iou'], 'MegaMulAdd=',scn.forward_pass_multiplyAdd_count/r['nmodels_sum']/1e6, 'MegaHidden',scn.forward_pass_hidden_states/r['nmodels_sum']/1e6,'time=',time.time() - start,'s')
      print(r['iou_all'])

  print('The end.')
