import os, sys, json, torch, copy, math
import numpy as np
import third_party.efficient_3DCNN.test as test
from torch import nn, optim
from torch.optim import lr_scheduler
from third_party.efficient_3DCNN.model import generate_model
from third_party.efficient_3DCNN.mean import get_mean, get_std
from third_party.efficient_3DCNN.spatial_transforms import *
from third_party.efficient_3DCNN.temporal_transforms import *
from third_party.efficient_3DCNN.target_transforms import ClassLabel, VideoID
from third_party.efficient_3DCNN.target_transforms import Compose as TargetCompose
from third_party.efficient_3DCNN.dataset import get_training_set, get_validation_set, get_test_set
from third_party.efficient_3DCNN.utils import *
from third_party.efficient_3DCNN.train import train_epoch
from third_party.efficient_3DCNN.validation import val_epoch
from third_party.thop.thop.profile import profile
from pruning.pytorch_snip.prune import pruning, do_statistics_model, dump_neuron_per_layer
from pruning_related import refine_model_classification, refine_model_I3D
from pruning.pytorch_snip.video_classification import create_network_connection_dict
from aux.utils import weight_init, remove_module_key
from tensorboardX import SummaryWriter
from configs import set_config


if __name__ == '__main__':
    opt = set_config()
    opt.prune_spatial_size = opt.sample_size
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.width_mult) + 'x',
                               opt.modality, str(opt.sample_duration)])
    opt.batch = opt.n_threads
    print(opt)
    torch.manual_seed(opt.manual_seed)
    model, parameters = generate_model(opt)
    weight_init(model, mode=opt.weight_init) if (not os.path.exists(opt.pretrain_path)) else None
    # print('Original model:', model)

    if not opt.no_cuda:
        model = model.cuda()

    # Loss
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            RandomHorizontalFlip(),
            #RandomRotate(),
            #RandomResize(),
            crop_method,
            #MultiplyValues(),
            #Dropout(),
            #SaltImage(),
            #Gaussian_blur(),
            #SpatialElasticDisplacement(),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data, batch_size=opt.batch, shuffle=True, num_workers=0,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer = optim.SGD(
            parameters, lr=opt.lr, momentum=opt.momentum, dampening=dampening,
            weight_decay=opt.weight_decay, nesterov=opt.nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)
    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        #temporal_transform = LoopPadding(opt.sample_duration)
        temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.downsample)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data, batch_size=16, shuffle=False, num_workers=0,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'prec1', 'prec5'])
    print('Trainset len: {}, validset len: {}'.format(len(train_loader.dataset), len(val_loader.dataset)))

    best_prec1 = 0
    opt.writer = SummaryWriter(opt.result_path)

    # Profile for full model Zhiwei
    grad_mode = 'raw' if opt.enable_raw_grad else 'abs'
    model_full = copy.deepcopy(model)
    profile_input = torch.randn(1, 3, opt.sample_duration,
                                opt.sample_size, opt.sample_size).cuda()
    flops_full, params_full, memory_full, resource_list = profile(model_full.cuda(),
                                                                  inputs=(profile_input,),
                                                                  verbose=False,
                                                                  resource_list_type=opt.resource_list_type)
    del model_full

    if opt.enable_neuron_prune:
        if opt.model == 'mobilenetv2':  # was in rebuttal previously
            json_path = 'data/ucf101/ucf101_{}.json'.format(opt.model)
            network_connection_dict = create_network_connection_dict(model, json_network_connection=json_path)
        else:
            json_path, network_connection_dict = None, None

        if os.path.exists(opt.pretrain_path):
            pretrain_name = opt.pretrain_path.split('/')[-1].split('.')[0]
            file_path = 'data/ucf101/ucf101_{}_sz{}_{}_{}_{}.npy'.format(
                opt.model, opt.prune_spatial_size,
                opt.weight_init, grad_mode, pretrain_name)
        else:
            file_path = 'data/ucf101/ucf101_{}_sz{}_{}_{}.npy'.format(
                opt.model, opt.prune_spatial_size,
                opt.weight_init, grad_mode)

        train_loader_pruning = torch.utils.data.DataLoader(
            training_data, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=True)

        outputs = pruning(file_path, model, train_loader_pruning, criterion, opt,
                          enable_3dunet=False, enable_hidden_sum=False, width=0,
                          resource_list=resource_list, network_name=opt.model)
        assert outputs[0] == 0
        neuron_mask_clean, hidden_mask = outputs[1], outputs[2]
        n_params_org, n_neurons_org = do_statistics_model(model)

        if opt.model == 'i3d':
            new_model = refine_model_I3D(model, neuron_mask_clean)
        else:
            new_model = refine_model_classification(model,
                                                    neuron_mask_clean,
                                                    opt.model,
                                                    network_connection_dict=network_connection_dict,
                                                    enable_raw_grad=opt.enable_raw_grad)

        # ==== Assign model parameters to new model
        # if not pretrained, do below; otherwise, keep parameters in the full model.
        weight_init(new_model, mode=opt.weight_init) if (not os.path.exists(opt.pretrain_path)) else None

        del model
        model = new_model.cpu()
        # print('New model:', model)

        n_params_refined, n_neurons_refined = do_statistics_model(model)
        print('Statistics, org, params: {}, neurons: {}; refined, params: {} ({:.4f}%), neurons: {} ({:.4f}%)' \
              .format(n_params_org,
                      n_neurons_org,
                      n_params_refined,
                      n_params_refined * 100 / n_params_org,
                      n_neurons_refined,
                      n_neurons_refined * 100 / n_neurons_org))

    # Calculate Flops, Params, Memory of New Model
    model_flops = copy.deepcopy(model)
    flops, params, memory, _ = profile(model_flops.cuda(),
                                       inputs=(profile_input,),
                                       verbose=False,
                                       resource_list_type=opt.resource_list_type)
    print('Full model, flops: {:.4f}G, params: {:.4f}MB, memory: {:.4f}MB' \
          .format(flops_full / 1e9, params_full * 4 / (1024 ** 2), memory_full * 4 / (1024 ** 2)))
    print('New model, flops: {:.4f}G, params: {:.4f}MB, memory: {:.4f}MB' \
          .format(flops / 1e9, params * 4 / (1024 ** 2), memory * 4 / (1024 ** 2)))
    del model_flops, profile_input

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch'].lower()
        best_prec1 = checkpoint['best_prec1']
        opt.begin_epoch = checkpoint['epoch']
        new_state_dict = remove_module_key(checkpoint['state_dict'])
        model.load_state_dict(new_state_dict)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

    # Reset to optimizer, otherwise no training
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.lr,
                          momentum=opt.momentum,
                          dampening=dampening,
                          weight_decay=opt.weight_decay,
                          nesterov=opt.nesterov)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)

    if opt.enable_train:
        for i in range(opt.begin_epoch, opt.epoch + 1):
            if not opt.no_train:
                adjust_learning_rate(optimizer, i, opt)
                train_epoch(i, train_loader, model, criterion, optimizer, opt,
                            train_logger, train_batch_logger)
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1
                    }
                save_checkpoint(state, False, opt)

            if not opt.no_val:
                validation_loss, prec1 = val_epoch(i, val_loader, model, criterion,
                                                   opt, val_logger)
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1
                    }
                save_checkpoint(state, is_best, opt)

    if opt.enable_test:
        if False:
            spatial_transform = Compose([
                Scale(int(opt.sample_size / opt.scale_in_test)),
                CornerCrop(opt.sample_size, opt.crop_position_in_test),
                ToTensor(opt.norm_value), norm_method
            ])
            # temporal_transform = LoopPadding(opt.sample_duration, opt.downsample)
            temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
            target_transform = VideoID()

            test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                     target_transform)
            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=16,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
            test.test(test_loader, model, opt, test_data.class_names)
        else:
            validation_loss, prec1 = val_epoch(0, val_loader, model, criterion,
                                               opt, val_logger)

    opt.writer.close()
