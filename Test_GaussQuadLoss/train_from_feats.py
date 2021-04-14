#!/usr/bin/env python
from __future__ import print_function
import argparse
import logging
import random
import time

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

from kaldi_feats_dataset import KaldiFeatsDataset, SpkrSampler, SpkrSplit
from xvector_model import Xvector9s, GaussQuadLoss, ComputeLoss
from utils import save_checkpoint, resume_checkpoint, AverageMeter, accuracy, load_model

logger = logging.getLogger(__name__)


def train(args, model, device, train_loader, optimizer, epoch, loss_fn):
    """
    Perform training of model
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # make sure we are in train mode (except frozen layers)
    model.train_with_freeze()

    end = time.time()
    for step, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.random_frame_size:
            # random segment truncation for batch
            frame_length = random.randint(args.min_frames, args.max_frames)
            data = data[:,:,0:frame_length]
                    
        # copy data to GPU
        data, target = data.to(device), target.to(device)

        # compute output and loss
        x, y, z, w = model(data)
        loss, acc = loss_fn(y, z, w, target)
        losses.update(loss.item(), data.size(0))
        top1.update(acc[0][0], data.size(0))

        # compute gradient and do step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update any generative parameters
        model.update_params(x, y, z, target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.log_interval == 0:
           logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
           with torch.no_grad():
               print("center loss ", model.PLDA.center_loss(x))
               print("norm loss ", model.PLDA.norm_loss(x))
               #print("mean loss ", model.output.mean_loss())
               print("Norm scale ", model.PLDA.norm_scale)
               print("d_ac ", model.PLDA.d_ac)

    # print epoch training loss
    logger.info('Train Epoch: [{0}]\t'
          'Time {batch_time.avg:.3f}s\t'
          'Loss {loss.avg:.3f}\t'
          'Prec@1 {top1.avg:.3f}'.format(
           epoch,  batch_time=batch_time,
           loss=losses, top1=top1))

def validate(args, model, device, val_loader, epoch, loss_fn):
    """
    Evaluate model on validation data
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # set model to eval mode
    model.eval()
    cost = 'test'

    with torch.no_grad():
        end = time.time()
        for step, (data, target) in enumerate(val_loader):

            if args.random_frame_size:
                # random segment truncation for batch
                frame_length = random.randint(args.min_frames, args.max_frames)
                data = data[:,:,0:frame_length]
                    
            # copy data to GPU
            data, target = data.to(device), target.to(device)

            # compute model output and loss
            x, y, z, w = model(data)
            loss, acc = loss_fn(y, z, w, target, eval_mode=True)
            losses.update(loss.item(), data.size(0))
            top1.update(acc[0][0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    logger.info('Valid Epoch: [{0}]\t'
          'Time {batch_time.avg:.3f}s\t'
          '{1} {loss.avg:.3f}\t'
          'Prec@1 {top1.avg:.3f}'.format(
           epoch,  cost, batch_time=batch_time,
           loss=losses, top1=top1))

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
                data = data[:,:,0:frame_length]
                    
            # copy data to GPU
            data, target = data.to(device), target.to(device)

            # compute model output and update PLDA
            x, y, z, w = model(data)
            model.PLDA.update_plda(y, target)

    logger.info("PLDA training epoch, count range %.2f to %.2f" % (model.PLDA.counts.min(), model.PLDA.counts.max()))


def main():
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    #logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

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
    parser.add_argument('--step-size', type=int, default=1, dest='step_size',
                        help='Epochs before learning rate drop (default: 1)')
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
    parser.add_argument('--length_norm', action='store_true', default=False,
                        help='length normalize embeddings')
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
    parser.add_argument('--train_cost', default='CE', type=str,
                        choices=['CE', 'GaussLoss'], help='training cost (default: CE)'),
    parser.add_argument('--duration_T0', type=float, default=0.0,
                        help='duration T0 in frames (default: 0.0 = no duration model)')
    parser.add_argument('--enroll_type', default='ML',
                        choices=['ML', 'MAP', 'Bayes', 'discr'], help='enrollment type (default: ML)')
    parser.add_argument('--cut_corr', type=float, default=0.0,
                        help='enrollment cut correlation (between 0 and 1)')
    parser.add_argument('--enroll_N0', type=int, default=9,
                        help='Gaussian enrollment number of cuts (default: 9)')
    parser.add_argument('--fixed_N', action='store_true', default=False,
                        help='fixed or variable number of Gaussian enrollment cuts (default False)')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    logger.info("Use CUDA: %s", use_cuda)

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
    train_dataset = KaldiFeatsDataset(args.feats_scp_filename, args.utt2spk_filename, num_frames=frame_length, cost=args.train_cost)

    logger.info("Creating test dataset using random speaker split (train: %f)" % (args.train_portion))
    train_dataset, test_dataset = SpkrSplit(train_dataset, args.train_portion)
    test_dataset.set_Gauss_cost(num_spkr_utt=10)

    if args.valid_feats_scp and args.valid_utt2spk:
        logger.info("Creating valid dataset")
        valid_dataset = KaldiFeatsDataset(args.valid_feats_scp, args.valid_utt2spk, num_frames=frame_length, spk2int=train_dataset.spk2int, cost=args.train_cost)

    if use_cuda:
        logger.info("Creating training data loaders with %d workers with minibatch %d", args.num_workers,args.batch_size)
    else:
        logger.info("Creating training data loader with minibatch %d",args.batch_size)
        torch.set_num_threads(8)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset,
                       batch_size=args.batch_size,
                       shuffle=False, sampler=SpkrSampler(train_dataset), **kwargs)

    batch_size=2
    logger.info("Creating validation data loader with minibatch %d",batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                       batch_size=batch_size,
                       shuffle=False, sampler=SpkrSampler(valid_dataset), **kwargs)

    logger.info("Creating test data loader with minibatch %d",args.test_batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                       batch_size=args.test_batch_size,
                       shuffle=False, sampler=SpkrSampler(test_dataset), **kwargs)

    logger.info("Creating model")
    model = Xvector9s(input_dim=args.feature_dim,
                      layer_dim=args.layer_dim,
                      embedding_dim=args.embedding_dim,
                      num_classes=len(train_dataset.spks),
                      T0=args.duration_T0,
                      length_norm=args.length_norm).to(device)

    logger.info("Model summary")
    summary(model, input_size=(args.feature_dim, 10000))

    if use_cuda and torch.cuda.device_count() > 1:
        logger.info("Setting up data parallel on %d devices", torch.cuda.device_count())
        model = torch.nn.DataParallel(model).to(device)

    logger.info("Creating optimizer %s with learning-rate %f", args.optimizer, args.learning_rate)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_epoch = 1

    train_loss = GaussQuadLoss(d=args.embedding_dim, M=len(train_dataset.spks), plda=model.plda, 
                               N0=args.enroll_N0, fixed_N=args.fixed_N,
                               r = args.cut_corr, enroll_type = args.enroll_type, ge2e=False).to(device)

    test_loss = GaussQuadLoss(plda=model.plda, r = args.cut_corr, enroll_type = args.enroll_type, ge2e=True).to(device)

    if args.resume_checkpoint:
        start_epoch, model, optimizer = resume_checkpoint(args.resume_checkpoint, model, optimizer, device)

    if args.load_model:
        model = load_model(args.load_model, model, device)
        if 0:
            logger.info("Freezing embedding layers of initial model...")
            model.freeze_embedding()
        elif 0:
            logger.info("Freezing prepooling layers of initial model...")
            model.freeze_prepooling()

    if args.step_decay > 0:
        logger.info("Creating step learning rate scheduler, dropping by %f every %d epochs", args.step_decay, args.step_size)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_decay)
    else:
        logger.info("Creating exponential learning rate scheduler, gamma of %f", -args.step_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=-args.step_decay)

    logger.info("Starting training from epoch %d for %d epochs", start_epoch, args.epochs)

    # Set up learning rate if starting from checkpoint
    for epoch in range(0, start_epoch+1):
        # step learning rate
        scheduler.step()

    vstate = None
    if 1:
        train_plda(args, model, device, train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        # step learning rate
        scheduler.step()

        # train an epoch
        if not args.valid_only:
            train(args, model, device, train_loader, optimizer, epoch, train_loss)

        # validate and test
        tstate = random.getstate()
        if vstate is None:
            vstate = random.getstate()
        random.setstate(vstate)
        validate(args, model, device, valid_loader, epoch, train_loss)
        validate(args, model, device, test_loader, epoch, test_loss)
        random.setstate(tstate)
        if args.valid_only:
            exit(0)

        # save checkpoint
        save_checkpoint(model, optimizer, epoch, args.checkpoint_dir)


if __name__ == '__main__':
    main()
