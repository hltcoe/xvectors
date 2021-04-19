#!/usr/bin/env python
from __future__ import print_function
import argparse
import logging
import random
import time

import torch
import torch.nn
import torch.optim as optim

from torchsummary import summary
from xvectors.kaldi_feats_dataset import KaldiFeatsDataset, SpkrSampler, spkr_split
from xvectors.xvector_model import Xvector9s, train_with_freeze
from xvectors.plda_lib import compute_loss
from xvectors.utils import save_checkpoint, resume_checkpoint, AverageMeter, accuracy, load_model, bn_momentum_adjust, \
    LinearUpDownLR

logger = logging.getLogger(__name__)


def train(args, model, device, train_loader, optimizer, epoch, cost='CE', boost=0):
    """
    Perform training of model
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # make sure we are in train mode (except frozen layers)
    train_with_freeze(model)

    end = time.time()
    for step, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.random_frame_size:
            # random segment truncation for batch
            frame_length = random.randint(args.min_frames, args.max_frames)
            data = data[:, :, 0:frame_length]

        # copy data to GPU
        data, target = data.to(device), target.to(device)

        # compute output and loss
        x, y, z, output, w = model(data, target)
        loss, nloss, acc = compute_loss(x, z, output, w, target, cost, model, boost)
        losses.update(nloss.item(), data.size(0))
        top1.update(acc[0][0], data.size(0))
        if w is not None:
            print(w.sum())

        # compute gradient and do step using un-normalized loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.log_interval == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                        'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                        'Normalized Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            epoch, step, len(train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, top1=top1))

    # print epoch training loss
    for i, param_group in enumerate(optimizer.param_groups):
        lr = float(param_group['lr'])
        break
    logger.info('Train Epoch: [{0}]\t'
                'Time {batch_time.avg:.3f}s\t'
                'Normalized Loss {loss.avg:.3f}\t'
                'Prec@1 {top1.avg:.3f}\t'
                'lr {lr:.6f}'.format(
                    epoch, batch_time=batch_time,
                    loss=losses, top1=top1, lr=lr))


def validate(args, model, device, val_loader, epoch, cost='GaussLoss'):
    """
    Evaluate model on validation data
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # set model to eval mode
    model.eval()

    if cost == 'GaussLoss':
        embedding_only = True
    else:
        embedding_only = False
    with torch.no_grad():
        end = time.time()
        for step, (data, target) in enumerate(val_loader):

            if args.random_frame_size:
                # random segment truncation for batch
                frame_length = random.randint(args.min_frames, args.max_frames)
                data = data[:, :, 0:frame_length]

            # copy data to GPU
            data, target = data.to(device), target.to(device)

            # compute model output and loss
            x, y, z, output, w = model(data, embedding_only=embedding_only)
            loss, nloss, acc = compute_loss(x, z, output, w, target, cost, model)
            losses.update(nloss.item(), data.size(0))
            top1.update(acc[0][0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    logger.info('Valid Epoch: [{0}]\t'
                'Time {batch_time.avg:.3f}s\t'
                'Norm {1} {loss.avg:.3f}\t'
                'Prec@1 {top1.avg:.3f}'.format(
                    epoch, cost, batch_time=batch_time,
                    loss=losses, top1=top1))

    return losses.avg


def train_plda(args, model, device, train_loader):
    """
    Train plda parameters with given embedding
    """
    # set model to eval mode
    model.eval()

    with torch.no_grad():
        for step, (data, target) in enumerate(train_loader):

            if args.random_frame_size:
                # random segment truncation for batch
                frame_length = random.randint(args.min_frames, args.max_frames)
                data = data[:, :, 0:frame_length]

            # copy data to GPU
            data, target = data.to(device), target.to(device)

            # compute model output and update PLDA
            x, y, z, output, w = model(data, embedding_only=True)
            model.PLDA.update_plda(y, target)

    logger.info("PLDA training epoch, count range %.2f to %.2f" % (model.PLDA.counts.min(), model.PLDA.counts.max()))


def main():
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    # logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch x-vectors')
    parser.add_argument('feats_scp_filename', help='train kaldi feats.scp')
    parser.add_argument('utt2spk_filename', help='train kaldi utt2spk')

    parser.add_argument('--valid-feats-scp', dest='valid_feats_scp', help='valid kaldi feats scp', default=None)
    parser.add_argument('--valid-utt2spk', dest='valid_utt2spk', help='valid kaldi utt2spk', default=None)

    parser.add_argument("--feature-dim", dest="feature_dim", default=23, type=int)
    parser.add_argument("--embedding-dim", dest="embedding_dim", default=256, type=int)
    parser.add_argument("--layer-dim", dest="layer_dim", default=256, type=int)
    parser.add_argument("--min-frames", dest="min_frames", default=200, type=int)
    parser.add_argument("--max-frames", dest="max_frames", default=400, type=int)
    parser.add_argument("--random-frame-size", dest="random_frame_size", default=False, action="store_true")

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=640, metavar='N',
                        help='input batch size for testing (default: 640)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=0.001, dest='learning_rate',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--step-size', type=int, default=0, dest='step_size',
                        help='Epochs before learning rate drop (default: 0)')
    parser.add_argument('--step-decay', type=float, default=0.1, dest='step_decay',
                        help='Learning rate drop factor (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0, metavar='M',
                        help='Optimizer weight decay (default: 0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S',
                        help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=4, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--checkpoint-dir', default='.', metavar='DIR',
                        help='directory to save checkpoints')
    parser.add_argument('--optimizer', dest='optimizer', default='sgd',
                        choices=['adam', 'sgd'], help='optimizer (default: sgd)')
    parser.add_argument('--num-workers', type=int, default=0, metavar='N',
                        help='num workers (default: 0)')
    parser.add_argument('--resume-checkpoint', dest='resume_checkpoint',
                        default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--load_model', dest='load_model',
                        default='', type=str, metavar='PATH',
                        help='path to initial model (default: none)')
    parser.add_argument('--train-portion', type=float, default=0.9, metavar='N',
                        help='train portion for random split (default: 0.9)')
    parser.add_argument('--valid_only', action='store_true', default=False,
                        help='validation only')
    parser.add_argument('--LLtype', dest='LLtype', default='linear',
                        choices=['linear', 'Gauss', 'Gauss_discr', 'xvec', 'None'],
                        help='log-likelihood output layer (default: linear)')
    parser.add_argument('--length_norm', action='store_true', default=False,
                        help='length normalize embeddings')
    parser.add_argument('--train_cost', default='CE', type=str,
                        choices=['CE', 'BCE', 'GaussLoss', 'BinLoss'], help='training cost (default: CE)'),
    parser.add_argument('--enroll_N0', type=int, default=9,
                        help='Gaussian enrollment number of cuts (default: 9)')
    parser.add_argument('--enroll_R', type=float, default=0.9,
                        help='Gaussian enrollment cut correlation (default: 0.9)')
    parser.add_argument('--enroll_type', dest='enroll_type', default='ML',
                        choices=['ML', 'MAP', 'Bayes'], help='Gaussian enrollment type (default: ML)')
    parser.add_argument('--fixed_N', action='store_true', default=False,
                        help='fixed or variable number of Gaussian enrollment cuts (default False)')
    parser.add_argument('--freeze_prepool', action='store_true', default=False,
                        help='freeze prepooling layers of initial model (default False)')
    parser.add_argument('--init_epochs', type=int, default=0, metavar='N',
                        help='number of epochs for initializer training (default: 0)')
    parser.add_argument('--init_learning_rate', type=float, default=0,
                        help='initializer learning rate (default: 0 implies normal learning rate)')
    parser.add_argument('--plda_learn_scale', type=float, default=0.01,
                        help='scale factor for sgd plda learning rate (default: 0.01)')
    parser.add_argument('--init_up', type=int, default=0, metavar='N',
                        help='number of epochs for linear ramp up stepsize (default: 0)')
    parser.add_argument('--train_boost', type=float, default=0,
                        help='training CE boost margin (default: 0)')
    parser.add_argument('--ResNet', action='store_true', default=False,
                        help='ResNet instead of TDNN (default False)')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    logger.info("Use CUDA: %s", use_cuda)
    torch.set_num_threads(8)

    logger.info("Setting random seed to %d", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info("Creating train dataset")
    if args.random_frame_size:
        # Random frame size: need to always read max then truncate later
        frame_length = args.max_frames
        logger.info("Random frame lengths from %d to %d, set default to read max", args.min_frames, args.max_frames)
    else:
        # Constant average frame size
        frame_length = (args.max_frames + args.min_frames) // 2
        logger.info("Fixed frame length %d", frame_length)
    train_dataset = KaldiFeatsDataset(args.feats_scp_filename, args.utt2spk_filename,
                                      num_frames=frame_length, cost=args.train_cost, enroll_N0=args.enroll_N0)

    test_dataset = None
    test_cost = 'GaussLoss'
    if args.train_cost == 'BCE':
        test_cost = 'BinLoss'
    if args.train_portion < 1.0:
        logger.info("Creating test dataset using random speaker split (train: %f)" % (args.train_portion))
        train_dataset, test_dataset = spkr_split(train_dataset, args.train_portion)
        if test_cost == 'BinLoss':
            test_dataset.set_Bin_cost()
        else:
            test_dataset.set_Gauss_cost()

    valid_dataset = None
    if args.valid_feats_scp and args.valid_utt2spk:
        logger.info("Creating valid dataset")
        valid_dataset = KaldiFeatsDataset(args.valid_feats_scp, args.valid_utt2spk, num_frames=frame_length,
                                          spk2int=train_dataset.spk2int, cost=args.train_cost)

    if use_cuda:
        logger.info("Creating training data loaders with %d workers with minibatch %d", args.num_workers,
                    args.batch_size)
    else:
        logger.info("Creating training data loader with minibatch %d", args.batch_size)
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False, sampler=SpkrSampler(train_dataset), **kwargs)
    batch_size = 2
    valid_loader = None
    if valid_dataset is not None:
        logger.info("Creating validation data loader with minibatch %d", batch_size)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False, sampler=SpkrSampler(valid_dataset, reset_flag=True),
                                                   **kwargs)
    test_loader = None
    if test_dataset is not None:
        logger.info("Creating test data loader with minibatch %d", args.test_batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.test_batch_size,
                                                  shuffle=False, sampler=SpkrSampler(test_dataset, reset_flag=True,
                                                                                     fixed_N=args.fixed_N), **kwargs)

    logger.info("Creating model")
    model_constructor_args = {
        'input_dim': args.feature_dim,
        'layer_dim': args.layer_dim,
        'embedding_dim': args.embedding_dim,
        'num_classes': len(train_dataset.spks),
        'LL': args.LLtype,
        'N0': args.enroll_N0,
        'fixed_N': args.fixed_N,
        'r': args.enroll_R,
        'enroll_type': args.enroll_type,
        'length_norm': args.length_norm,
        'resnet_flag': args.ResNet
    }
    model = Xvector9s(**model_constructor_args)

    model = model.to(device)
    logger.info("Model summary")
    summary(model, (args.feature_dim, 10000), device=device)

    if args.optimizer == 'sgd':
        logger.info("Creating optimizer %s with learning-rate %f, momentum %f, weight_decay %f", args.optimizer,
                    args.learning_rate, args.momentum, args.weight_decay)

        # Set learning rate for PLDA parameters (mainly scale factor) to be slower
        params_not_plda = [{'params': model.embed.parameters()}]
        if model.output is not None:
            params_not_plda.append({'params': model.output.parameters()})
        model_par = params_not_plda + [
            {'params': model.plda.parameters(), 'lr': args.plda_learn_scale * args.learning_rate}]
        optimizer = optim.SGD(model_par, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        if args.init_epochs and args.init_learning_rate:
            logger.info("Creating initial optimizer %s with learning-rate %f", args.optimizer, args.init_learning_rate)
            params_not_plda = [{'params': model.embed.parameters()}]
            if model.output is not None:
                params_not_plda.append({'params': model.output.parameters()})
            model_par = params_not_plda + [
                {'params': model.plda.parameters(), 'lr': args.plda_learn_scale * args.init_learning_rate}]
            init_optimizer = optim.SGD(model_par, lr=args.init_learning_rate, momentum=args.momentum,
                                       weight_decay=args.weight_decay)
        else:
            logger.info("Initial optimizer is regular one.")
            init_optimizer = optimizer

        # Log stepsizes
        for i, param_group in enumerate(optimizer.param_groups):
            lr = float(param_group['lr'])
            logger.info("Stepsize for group %f" % (lr))
        for i, param_group in enumerate(init_optimizer.param_groups):
            lr = float(param_group['lr'])
            logger.info("Stepsize for init group %f" % (lr))

    elif args.optimizer == 'adam':
        logger.info("Creating optimizer %s with learning-rate %f, weight_decay %f", args.optimizer, args.learning_rate,
                    args.weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        if args.init_epochs and args.init_learning_rate:
            logger.info("Creating initial optimizer %s with learning-rate %f", args.optimizer, args.init_learning_rate)
            init_optimizer = optim.Adam(model.parameters(), lr=args.init_learning_rate, weight_decay=args.weight_decay)
        else:
            logger.info("Initial optimizer is regular one.")
            init_optimizer = optimizer

    if args.load_model:
        model = load_model(args.load_model, device)
        if args.freeze_prepool:
            logger.info("Freezing prepooling layers of initial model...")
            model.freeze_prepooling()

    start_epoch = 1

    if args.init_up:
        N0 = args.init_up
        logger.info("Creating linear up/down learning rate scheduler, up %d then decay to %d", N0, args.epochs)
        scheduler = LinearUpDownLR(optimizer, N0, args.epochs)
    elif args.step_size == 0:
        logger.info("Creating validation error step learning rate scheduler, dropping by %f", args.step_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.step_decay, patience=10, verbose=True)
    elif args.step_decay > 0:
        logger.info("Creating step learning rate scheduler, dropping by %f every %d epochs", args.step_decay,
                    args.step_size)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_decay)
    else:
        logger.info("Creating exponential learning rate scheduler, gamma of %f", -args.step_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=-args.step_decay)

    init_scheduler = None
    if args.init_epochs and args.init_learning_rate:
        max_lr = args.learning_rate
        base_lr = args.init_learning_rate
        gamma = (max_lr / base_lr) ** (1.0 / args.init_epochs)
        logger.info("Creating initial exponential learning rate increase scheduler, gamma of %f", gamma)
        init_scheduler = optim.lr_scheduler.ExponentialLR(init_optimizer, gamma)

    if args.resume_checkpoint:
        start_epoch, model, optimizer, scheduler = resume_checkpoint(args.resume_checkpoint, model, optimizer,
                                                                     scheduler, device)

    # Validation only
    if args.valid_only:
        # validate and test (with initial random seed)
        epoch = start_epoch
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if valid_loader is not None:
            validate(args, model, device, valid_loader, epoch, args.train_cost)
        validate(args, model, device, test_loader, epoch, test_cost)
        exit(0)

    # Initial training: no scheduler or validation
    if start_epoch <= args.init_epochs:
        if init_scheduler is None and start_epoch == 1:
            model.loo_flag = False  # Cold start can't use leave-one-out
            logger.info(" turning off leave-one-out for initialization")
        logger.info("Starting initializer training from epoch %d for %d epochs", start_epoch, args.init_epochs)
        for epoch in range(start_epoch, args.init_epochs + 1):
            # train an epoch
            train(args, model, device, train_loader, init_optimizer, epoch, args.train_cost, args.train_boost)
            if not model.loo_flag:
                model.loo_flag = True
                logger.info(" turning leave-one-out back on")

            # step learning rate
            if init_scheduler is not None:
                init_scheduler.step()
                # save checkpoint
                save_checkpoint(model, model_constructor_args, init_optimizer, init_scheduler, epoch,
                                args.checkpoint_dir)
            else:
                # save checkpoint
                save_checkpoint(model, model_constructor_args, init_optimizer, scheduler, epoch, args.checkpoint_dir)

        start_epoch = args.init_epochs + 1

    # Training
    best_loss = None
    logger.info("Starting training from epoch %d for %d epochs", start_epoch, args.epochs)
    for epoch in range(start_epoch, args.epochs + 1):

        # train an epoch
        train(args, model, device, train_loader, optimizer, epoch, args.train_cost, args.train_boost)

        # validate and test (with initial random seed)
        tstate = random.getstate()
        random.seed(args.seed)
        loss = 1.0
        if valid_loader is not None:
            loss = validate(args, model, device, valid_loader, epoch, args.train_cost)
        if test_loader is not None:
            loss = validate(args, model, device, test_loader, epoch, test_cost)
        random.setstate(tstate)

        # step learning rate
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss)
        else:
            scheduler.step()

        # adjust batch-norm momemtum also
        # bn_momentum_adjust(model, optimizer)

        # check best validation loss or reset training (disabled)
        best_flag = False

        # save checkpoint
        save_checkpoint(model, model_constructor_args, optimizer, scheduler, epoch, args.checkpoint_dir, best_flag)


if __name__ == '__main__':
    main()
