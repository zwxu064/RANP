import torch
from torch.autograd import Variable
import time
import os
import sys
from third_party.efficient_3DCNN.utils import *
import torch.nn.functional as F


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    data_loader_len = len(data_loader)
    end_time = time.time()

    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = model(inputs)

        # Zhiwei don't need loc_loss and cls_loss in train_i3d as they are just
        # by upsample from batch*101*1 to batch*101*num_frame, so use the same
        # as mobilenetv2
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'prec1': top1.val,
            'prec5': top5.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                      epoch,
                      i,
                      data_loader_len,
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                      lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'prec1': top1.avg,
        'prec5': top5.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    opt.writer.add_scalar('Train/loss', losses.avg, epoch)
    opt.writer.add_scalar('Train/prec1', top1.avg, epoch)
    opt.writer.add_scalar('Train/prec5', top5.avg, epoch)

    #if epoch % opt.checkpoint == 0:
    #    save_file_path = os.path.join(opt.result_path,
    #                                  'save_{}.pth'.format(epoch))
    #    states = {
    #        'epoch': epoch + 1,
    #        'arch': opt.arch,
    #        'state_dict': model.state_dict(),
    #        'optimizer': optimizer.state_dict(),
    #    }
    #    torch.save(states, save_file_path)
