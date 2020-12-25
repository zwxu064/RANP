import argparse
import os


def set_config():
  parser = argparse.ArgumentParser(description='configures of project')
  parser.add_argument('--batch', type=int, default=None, help='batch size in training')
  parser.add_argument('--dataset', default='shapenet', type=str, help="shapenet, brats, ucf101")
  parser.add_argument('--data_dir', type=str, default='dataset', help='data directory')
  parser.add_argument('--spatial_size', type=int, default=None, help='spatial size')
  parser.add_argument('--valid_spatial_size', type=int, default=64, help='valid spatial size')
  parser.add_argument('--scale', type=int, default=1, help='scale')
  parser.add_argument('--epoch', type=int, default=None, help='epoch')
  parser.add_argument('--resume_epoch', type=int, default=-1, help='resume train epoch')
  parser.add_argument('--test_epoch', type=int, default=-1, help='resume test epoch')
  parser.add_argument('--lr', type=float, default=None, help='lr')
  parser.add_argument('--lr_decay', type=float, default=None, help='lr decay')
  parser.add_argument('--w_decay', type=float, default=1e-4, help='weight decay')
  parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
  parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='checkpoint directory')
  parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
  parser.add_argument('--enable_train', action='store_true', default=False, help='enable train')
  parser.add_argument('--enable_test', action='store_true', default=False, help='enable test')
  parser.add_argument('--enable_cuda', action='store_true', default=True, help='enable cuda')
  parser.add_argument('--enable_bias', action='store_true', default=False, help='enable bias')
  parser.add_argument('--weight_init', type=str, default='xn', help='weight initialization', choices=['xn', 'ort'])

  parser.add_argument('--neuron_sparsity', type=float, default=None, help='neuron sparsity')
  parser.add_argument('--hidden_layer_sparsity', type=float, default=0, help='hidden layer sparsity')
  parser.add_argument('--param_sparsity', type=float, default=0, help='SNIP param pruning')
  parser.add_argument('--layer_sparsity_list', type=float, default=0, help='layer wise neuron pruning')

  parser.add_argument('--acc_mode', type=str, default='sum', help='mode of mask grads, mean, max, sum')
  parser.add_argument('--prune_spatial_size', type=int, default=64, help='prune spatial size')
  parser.add_argument('--number_of_fmaps', type=int, default=4, help='depth of network')
  parser.add_argument('--disable_deepmodel_pooling', action='store_true', default=False,
                      help='disable max pooling for deep model (number of fmaps more than 4), needs least spatial size')
  parser.add_argument('--width', type=int, default=2, help='number of layers in an encoder/decoder')
  parser.add_argument('--res_type', type=str, default=None, help='None, layer, or block')
  parser.add_argument('--local_rank', type=int, help='local rank for GPUs')
  parser.add_argument('--enable_hard_padding', action='store_true', default=False, help='hard padding for sz > 96')
  parser.add_argument('--enable_raw_grad', action='store_true', default=False, help='enable raw gradient values for neuron importance')

  parser.add_argument('--random_method', type=int, default=None, choices=[0, 1, None], help='random method 0 or 1')
  parser.add_argument('--random_sparsity', type=float, default=0, help='random sparsity neuron pruning')
  parser.add_argument('--random_sparsity_seed', type=int, default=0, help='random sparsity seed for neuron pruning')
  parser.add_argument('--enable_layer_neuron_display', action='store_true', default=False, help='display layer neuron')
  parser.add_argument('--resource_list_type', type=str, default='grad_flops',
                      choices=['vanilla', 'grad', 'param', 'flops', 'memory', 'grad_param', 'grad_flops', 'grad_memory'],
                      help='use resource list for layer balance')
  parser.add_argument('--resource_list_lambda', type=float, default=None, help='control the importance of resource list weights')
  parser.add_argument('--valid_min_epoch', type=int, default=80, help='min epoch to do validation to save time')
  parser.add_argument('--load_transfer_model_path', type=str, default='', help='load model from BraTS')
  parser.add_argument('--save_transfer_model_path', type=str, default='', help='save model for BraTS')
  parser.add_argument('--enable_ssc_unet', action='store_true', default=False)
  parser.add_argument('--enable_viz', action='store_true', default=False, help='visualization')

  # For BraTS
  parser.add_argument('--years', nargs='+', type=int, default=2018, help='BraTS year')
  parser.add_argument('--alpha', type=float, default=0.01,
                      help='weight for cross entropy, others for dice')
  parser.add_argument('--ignore_index', type=int, default=255,
                      help='ignore index of ground truth')

  # For UCF101
  parser.add_argument('--lr_steps', default=None, type=float, nargs="+", metavar='LRSteps')
  parser.add_argument('--video_path', default='video_kinetics_jpg', type=str, help='Directory path of Videos')
  parser.add_argument('--annotation_path', default='kinetics.json', type=str, help='Annotation file path')
  parser.add_argument('--checkpoifnt_dir', default='checkpoints', type=str, help='Result directory path')
  parser.add_argument('--store_name', default='model', type=str, help='Name to store checkpoints')
  parser.add_argument('--modality', default='RGB', type=str, help='Modality of input data. RGB, Flow or RGBFlow')
  parser.add_argument('--n_finetune_classes', default=400, type=int,
                      help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
  parser.add_argument('--sample_size', default=None, type=int, help='Height and width of inputs')
  parser.add_argument('--downsample', default=1, type=int, help='Downsampling. Selecting 1 frame out of N')
  parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
  parser.add_argument('--n_scales', default=5, type=int, help='Number of scales for multiscale cropping')
  parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
  parser.add_argument('--train_crop', default='corner', type=str,
                      help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
  parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
  parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
  parser.add_argument('--mean_dataset', default='activitynet', type=str,
                      help='dataset for mean values of mean subtraction (activitynet | kinetics)')
  parser.add_argument('--no_mean_norm', action='store_true', help='If true, inputs are not normalized by mean.')
  parser.set_defaults(no_mean_norm=False)
  parser.add_argument('--std_norm', action='store_true', default=False, help='If true, inputs are normalized by standard deviation.')
  parser.add_argument('--nesterov', action='store_true', default=False, help='Nesterov momentum')
  parser.add_argument('--lr_patience', default=10, type=int,
                      help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
  parser.add_argument('--begin_epoch', default=1, type=int,
                      help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
  parser.add_argument('--n_val_samples', default=3, type=int, help='Number of validation samples for each activity')
  parser.add_argument('--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
  parser.add_argument('--ft_portion', default='complete', type=str,
                      help='The portion of the model to apply fine tuning, either complete or last_layer')
  parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
  parser.set_defaults(no_train=False)
  parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
  parser.set_defaults(no_val=False)
  parser.add_argument('--test', action='store_true', help='If true, test is performed.')
  parser.set_defaults(test=False)
  parser.add_argument('--test_subset', default='val', type=str, help='Used subset in test (val | test)')
  parser.add_argument('--scale_in_test', default=1.0, type=float, help='Spatial scale in test')
  parser.add_argument('--crop_position_in_test', default='c', type=str,
                      help='Cropping method (c | tl | tr | bl | br) in test')
  parser.add_argument('--no_softmax_in_test', action='store_true',
                      help='If true, output for each clip is not normalized using softmax.')
  parser.set_defaults(no_softmax_in_test=False)
  parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
  parser.set_defaults(no_cuda=False)
  parser.add_argument('--n_threads', default=16, type=int, help='Number of threads for multi-thread loading')
  parser.add_argument('--checkpoint', default=10, type=int, help='Trained model is saved at every this epochs.')
  parser.add_argument('--no_hflip', action='store_true', help='If true holizontal flipping is not performed.')
  parser.set_defaults(no_hflip=False)
  parser.add_argument('--norm_value', default=1, type=int,
                      help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
  parser.add_argument('--model', default='resnet', type=str,
                      help='(resnet | preresnet | wideresnet | resnext | densenet | ')
  parser.add_argument('--version', default=1.1, type=float, help='Version of the model')
  parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
  parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
  parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
  parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
  parser.add_argument('--groups', default=3, type=int, help='The number of groups at group convolutions at conv layers')
  parser.add_argument('--width_mult', default=1.0, type=float,
                      help='The applied width multiplier to scale number of filters')
  parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
  parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
  parser.add_argument('--sample_duration', default=None, type=int, help='Temporal duration of inputs')

  args = parser.parse_args()

  #
  args.enable_deepmodel_pooling = not args.disable_deepmodel_pooling
  args.model = args.model.lower()
  args.dataset = args.dataset.lower()

  if args.dataset.find('shapenet') > -1:
    args.lr = 0.1 if args.lr is None else args.lr
    args.epoch = 200 if args.epoch is None else args.epoch
  elif args.dataset.find('brats') > -1:
    args.lr = 0.001 if args.lr is None else args.lr
    args.epoch = 200 if args.epoch is None else args.epoch

  args.checkpoint_dir = os.path.join(args.checkpoint_dir,
                                     '{}_{}'.format(args.dataset, args.model))
  os.makedirs(args.checkpoint_dir) if not os.path.exists(args.checkpoint_dir) else None
  args.device = 'cuda' if args.enable_cuda else 'cpu'

  # Auto config
  num_classes = {'shapenet': 50, 'brats': 5, 'ucf101': 101}
  batch = {'shapenet': 4, 'brats': 1, 'ucf101': 8}
  spatial_size = {'shapenet': 64, 'brats': 128, 'ucf101': None}
  lr = {'shapenet': 0.1, 'brats': 0.001, 'ucf101': 0.1}
  epoch = {'shapenet': 200, 'brats': 200, 'ucf101': 250}
  lr_decay = {'shapenet': 0.04, 'brats': 0.04, 'ucf101': 0}
  lr_steps = {'mobilenetv2': [40, 55, 65, 70, 200, 250],
              'i3d': [50, 100, 150, 200]}
  resource_list_lambda = {'shapenet': 11, 'brats': 15, 'ucf101': 80}
  neuron_sparsity = {'shapenet': 0.7824, 'brats': 0.7817}
  neuron_sparsity_ucf101 = {'mobilenetv2': 0.3315, 'i3d': 0.2532}
  sample_size_ucf101 = {'mobilenetv2': 112, 'i3d': 224}
  sample_duration = {'shapenet': 0, 'brats': 0, 'ucf101': 16}
  scale = {'shapenet': 10, 'brats': 1, 'ucf101': 16}

  args.n_class = num_classes[args.dataset]
  args.batch = batch[args.dataset] if args.batch is None else args.batch
  args.sptial_size = spatial_size[args.dataset] if args.spatial_size is None else args.spatial_size
  args.lr = lr[args.dataset] if args.lr is None else args.lr
  args.epoch = epoch[args.dataset] if args.epoch is None else args.epoch
  args.lr_decay = lr_decay[args.dataset] if args.lr_decay is None else args.lr_decay
  args.sample_duration = sample_duration[args.dataset] if args.sample_duration is None else args.sample_duration
  args.scale = scale[args.dataset] if args.scale is None else args.scale
  args.resource_list_lambda = resource_list_lambda[args.dataset] if args.resource_list_lambda is None else args.resource_list_lambda

  if args.dataset == 'ucf101':
    args.lr_steps = lr_steps[args.model] if args.lr_steps is None else args.lr_steps
    args.sample_size = sample_size_ucf101[args.model] if args.sample_size is None else args.sample_size

  args.dim = args.spatial_size
  assert not (args.enable_train and args.enable_test)

  if args.dataset in {'shapenet', 'brats'}:
    assert args.spatial_size >= 2 ** args.number_of_fmaps
    assert args.prune_spatial_size >= 2 ** args.number_of_fmaps

  args.enable_neuron_prune = (args.neuron_sparsity > 0) or (args.layer_sparsity_list > 0) or (args.random_sparsity > 0)
  args.enable_hidden_layer_prune = (args.hidden_layer_sparsity > 0)
  args.enable_param_prune = (args.param_sparsity > 0)
  assert (not args.enable_raw_grad) if args.enable_param_prune else True

  return args