import time
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numbers
import random
from tensorboardX import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint
from network import *
from dataloader import *
from prune import *
from prune_utils import *
from flops import *
from snip import SNIP


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Data options
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset to use')
    # Model options
    parser.add_argument('--network', type=str, default='lenet5', help='network architecture to use')
    parser.add_argument('--param_sparsity', type=float, default=0.9, help='level of param sparsity to achieve')
    parser.add_argument('--neuron_sparsity', type=float, default=0.9, help='level of neuron sparsity to achieve')
    parser.add_argument('--channel_sparsity', type=float, default=0.9, help='level of channel sparsity to achieve')
    # Train options
    parser.add_argument('--batch', type=int, default=100, help='number of examples per mini-batch')
    parser.add_argument('--epochs', type=int, default=250, help='number of training epochs')
    parser.add_argument('--optimizer', type=str, default='momentum', help='optimizer of choice')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='learning rate decay ratio')
    parser.add_argument('--lr_decay_step', type=int, default=25e3, help='learning rate decay step')
    parser.add_argument('--lr', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay ratio')
    parser.add_argument('--log_interval', type=int, default=20, help='log saving frequency')
    parser.add_argument('--seed_list', nargs='+', default=0, help='seeds')
    parser.add_argument('--relative_dir', type=str, default='.', help='relative directory')
    parser.add_argument('--weight_init', type=str, default='xn', help='xn, xu, kn, ku, orthogonal')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='save model dir')
    parser.add_argument('--resume_epoch', type=int, default='0', help='resume model')
    # Operations
    parser.add_argument('--enable_flops', action='store_true', default=False, help='enable flops')
    parser.add_argument('--enable_bias', action='store_true', default=True, help='enable bias')
    parser.add_argument('--enable_dump', action='store_true', default=False, help='dump for MatLab')
    parser.add_argument('--enable_param_prune', action='store_true', default=False, help='prune params')
    parser.add_argument('--enable_neuron_prune', action='store_true', default=False, help='prune neurons')
    parser.add_argument('--enable_channel_prune', action='store_true', default=False, help='prune channels')
    parser.add_argument('--enable_dump_features', action='store_true', default=False, help='dump features')
    # For channel prune options
    parser.add_argument('--acc_mode', type=str, default='mean', help='accumulation for importance of a channel')
    parser.add_argument('--norm', type=str, default='max', help='normalization over grads')
    args = parser.parse_args()
    args.device = device
    args.enable_cuda = device == 'cuda'
    args.n_class = 10

    if isinstance(args.seed_list, numbers.Number):
      args.seed_list = [args.seed_list]

    if args.enable_neuron_prune:
      args.log_dir_comment = 'neuron_prune'
    elif args.enable_channel_prune:
      args.log_dir_comment = 'channel_prune'
    else:
      args.log_dir_comment = 'param_prune'

    return args


def mnist_experiment(args):
  network_name = args.network.lower()
  if network_name == 'lenet300':
    net = LeNet_300_100(enable_bias=args.enable_bias)
  elif network_name == 'lenet5':
    net = LeNet_5(enable_bias=args.enable_bias)
  elif network_name == 'lenet5_caffe':
    net = LeNet_5_Caffe(enable_bias=args.enable_bias)
  else:
    assert False

  net = net.to(args.device)
  train_batch_size = val_batch_size = args.batch
  train_loader, val_loader = get_mnist_dataloaders(train_batch_size, val_batch_size,
                                                   args, enable_train_shuffle=True)

  return net, train_loader, val_loader


def cifar10_experiment(args):
  network_name = args.network.lower()
  if network_name == 'alexnet_v1':
    net = AlexNet(k=1, enable_bias=args.enable_bias)
  elif network_name == 'alexnet_v2':
    net = AlexNet(k=2, enable_bias=args.enable_bias)
  elif network_name == 'vgg_c':
    net = VGG('C', enable_bias=args.enable_bias, enable_dump_features=args.enable_dump_features)
  elif network_name == 'vgg_d':
    net = VGG('D', enable_bias=args.enable_bias)
  elif network_name == 'vgg_like':
    net = VGG('like', enable_bias=args.enable_bias)
  else:
    assert False

  net = net.to(args.device)
  train_batch_size = val_batch_size = args.batch
  train_loader, val_loader = get_cifar10_dataloaders(train_batch_size, val_batch_size, args,
                                                     enable_train_shuffle=True,
                                                     enable_train_trans=True)

  return net, train_loader, val_loader


def train(args):
    writer = SummaryWriter(log_dir='{}/runs/{}/{}_{}_seed{}_winit{}' \
                           .format(args.relative_dir, args.log_dir_comment, args.network,
                                   time.strftime("%Y%m%d_%H%M%S"), args.seed, args.weight_init))

    if args.dataset == 'mnist':
      net, train_loader, val_loader = mnist_experiment(args)
      train_loader_prune, _ = get_mnist_dataloaders(args.batch, args.batch,
                                                    args, enable_train_shuffle=False)
    elif args.dataset == 'cifar10':
      net, train_loader, val_loader = cifar10_experiment(args)
      train_loader_prune, _ = get_cifar10_dataloaders(args.batch, args.batch,
                                                      args, enable_train_shuffle=False,
                                                      enable_train_trans=False)
    else:
      assert False

    enable_nesterov = True if (args.optimizer == 'nesterov') else False

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=enable_nesterov)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, gamma=args.lr_decay_ratio)  # Zhiwei paper 0.1

    # Resume
    resume_model_path = os.path.join(args.checkpoint_dir, '{}_model_{}.pth'.format(args.network, args.resume_epoch))
    resume_opt_path = os.path.join(args.checkpoint_dir, '{}_optimizer_{}.pth'.format(args.network, args.resume_epoch))
    if os.path.exists(resume_model_path):
      checkpoint = torch.load(resume_model_path, map_location=lambda storage, loc: storage)
      net.load_state_dict(checkpoint)
    else:
      if args.resume_epoch > 0:
        print('{} not exist'.format(resume_model_path))

    if os.path.exists(resume_opt_path):
      checkpoint = torch.load(resume_opt_path, map_location=lambda storage, loc: storage)
      optimizer.load_state_dict(checkpoint)
    else:
      if args.resume_epoch > 0:
        print('{} not exist'.format(resume_opt_path))

    # Pre-training pruning using SKIP
    if True:
      criterion = F.nll_loss
      prune_mask_clean, _ = pruning(None, net, train_loader_prune, criterion, args,
                                    enable_kernel_mask=True, enable_hidden_mask=False,
                                    enable_3dunet=False, enable_hidden_sum=False)
    else:
      grads = SNIP(net, train_loader_prune, args)

      print('Check grads value for seeds', torch.stack([grad.abs().sum() for grad in grads]).sum())

      # Removing redundant and calculate gflops
      all_one_mask = param_prune(grads, param_sparsity=0)

      if args.enable_flops:
        flop_input, flop_target = next(iter(train_loader_prune))
        flop_input = flop_input.to(args.device)
        flops_no_prune, params_no_prune = cal_flops(net, all_one_mask, flop_input, comment='(no prune)')

      # Case 1: Parameter prune, has redundant retains
      if args.enable_param_prune:
        prune_mask = param_prune(grads, param_sparsity=args.param_sparsity)
        prune_mask_clean = remove_redundant(prune_mask, prune_mode='param')
        do_statistics(prune_mask, prune_mask_clean)

        # Flop of param prune before and after removing redundant params
        if args.enable_flops:
          flops_param_prune, params_param_prune = cal_flops(net, prune_mask, flop_input)
          flops_param_prune_clean, params_param_prune_clean = cal_flops(net, prune_mask_clean, flop_input)
          print("GFlops, param prune, original:{:.4f} (param:{:.0f}), pruned:{:.4f} ({:.4f}%, param:{:.0f}), "
                "clean:{:.4f} ({:.4f}%, param:{:.0f}).\n" \
                .format(flops_no_prune, params_no_prune,
                        flops_param_prune, flops_param_prune * 100 / flops_no_prune, params_param_prune,
                        flops_param_prune_clean, flops_param_prune_clean * 100 / flops_no_prune, params_param_prune_clean))

      # Case 2: neuron prune, not good, a whole layer will be removed
      if args.enable_neuron_prune:
        prune_mask = neuron_prune(grads, neuron_sparsity=args.neuron_sparsity, acc_mode=args.acc_mode)
        prune_mask_clean = remove_redundant(prune_mask, prune_mode='neuron')
        do_statistics(prune_mask, prune_mask_clean)

        if args.enable_flops:
          # Flop of neuron prune before and after removing redundant params
          flops_neuron_prune, params_neuron_prune = cal_flops(net, prune_mask, flop_input)
          flops_neuron_prune_clean, params_neuron_prune_clean = cal_flops(net, prune_mask_clean, flop_input)
          print("GFlops, neuron prune, original:{:.4f} (param:{:.0f}), pruned:{:.4f} ({:.4f}%, param:{:.0f}),"
                "clean:{:.4f} ({:.4f}%, param:{:.0f}).\n" \
                .format(flops_no_prune, params_no_prune,
                        flops_neuron_prune, flops_neuron_prune * 100 / flops_no_prune, params_neuron_prune,
                        flops_neuron_prune_clean, flops_neuron_prune_clean * 100 / flops_no_prune, params_neuron_prune_clean))

      # Case 3: good, but pay attention to acc_mode='max'
      if args.enable_channel_prune:
        prune_mask = channel_prune(grads, channel_sparsity=args.channel_sparsity,
                                   acc_mode=args.acc_mode, norm=args.norm)
        prune_mask_clean = remove_redundant(prune_mask, prune_mode='channel')
        do_statistics(prune_mask, prune_mask_clean)

        if args.enable_flops:
          # Flop of channel prune before and after removing redundant params
          flops_channel_prune, params_channel_prune = cal_flops(net, prune_mask, flop_input)
          flops_channel_prune_clean, params_channel_prune_clean = cal_flops(net, prune_mask_clean, flop_input)
          print("GFlops, channel prune, original:{:.4f} (param:{:.0f}), pruned:{:.4f} ({:.4f}%, param:{:.0f}),"
                "clean:{:.4f} ({:.4f}%, param:{:.0f}).\n" \
                .format(flops_no_prune, params_no_prune,
                        flops_channel_prune, flops_channel_prune * 100 / flops_no_prune, params_channel_prune,
                        flops_channel_prune_clean, flops_channel_prune_clean * 100 / flops_no_prune, params_channel_prune_clean))

      if args.enable_channel_prune:
        # Case 4: message passing prune + remove directly
        mp_mask = message_passing_prune(grads, channel_sparsity=args.channel_sparsity,
                                        penalty=10, acc_mode=args.acc_mode, norm=args.norm)
        do_statistics(prune_mask, mp_mask)
        print('=> MP and channel prune clean cmp: {}'.format(check_same(mp_mask, prune_mask_clean)))
        for jj in range(len(mp_mask)):
          if not torch.equal(mp_mask[jj], prune_mask_clean[jj]):
            print('AW', len(mp_mask), mp_mask[jj].size(), jj, mp_mask[jj].flatten().sum(), prune_mask_clean[jj].flatten().sum())

      # Dump and train
      dump_grad_mask(grads, prune_mask, args) if args.enable_dump else None

    apply_prune_mask(net, prune_mask_clean)

    trainer = create_supervised_trainer(net, optimizer, F.nll_loss, args.device)
    evaluator = create_supervised_evaluator(net, {'accuracy': Accuracy(), 'nll': Loss(F.nll_loss)}, args.device)

    pbar = ProgressBar()
    pbar.attach(trainer)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        lr_scheduler.step()
        iter_in_epoch = (engine.state.iteration - 1) % len(train_loader) + 1
        if engine.state.iteration % args.log_interval == 0:
            # pbar.log_message("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
            #       "".format(engine.state.epoch, iter_in_epoch, len(train_loader), engine.state.output))
            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch(engine):
        evaluator.run(val_loader)

        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']

        # pbar.log_message("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        #       .format(engine.state.epoch, avg_accuracy, avg_nll))

        writer.add_scalar("validation/loss", avg_nll, engine.state.iteration)
        writer.add_scalar("validation/accuracy", avg_accuracy, engine.state.iteration)

    # Save models
    handler = ModelCheckpoint(args.checkpoint_dir, args.network, save_interval=args.epochs, n_saved=1, create_dir=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'model': net, 'optimizer': optimizer})

    if args.resume_epoch < args.epochs:
      trainer.run(train_loader, args.epochs)
    else:
      net.train()
      for idx, data in enumerate(train_loader_prune):
        if idx == 0:
          net(data[0].cuda(), epoch_id=args.epochs, batch_id=idx+1, gt=data[1])

    # Let's look at the final weights
    # for name, param in net.named_parameters():
    #     if name.endswith('weight'):
    #         writer.add_histogram(name, param)

    writer.close()
    print('Finish training!')


if __name__ == '__main__':
    args = parse_arguments()

    for seed in args.seed_list:
        seed = int(seed)

        print('=' * 80)
        print('==== network:{}, dataset:{}, lr:{}, epochs:{},\n'
              '==== param sparsity:{}, neuron sparsity:{}, channel sparsity:{},\n'
              '==== optimizer:{}, lr_decay_step:{}, lr_decay_ratio:{},\n'
              '==== weight_decay:{}, seed:{}, init:{}' \
              .format(args.network, args.dataset, args.lr, args.epochs,
                      args.param_sparsity, args.neuron_sparsity,
                      args.channel_sparsity, args.optimizer, args.lr_decay_step,
                      args.lr_decay_ratio, args.weight_decay, seed, args.weight_init))
        print('=' * 80, '\n')

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        args.seed = seed
        train(args)
