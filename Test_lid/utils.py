from __future__ import print_function
import os
import torch
import logging
import torch.nn

import torch.optim.lr_scheduler

logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir, best_flag=False):
    """
    Saving checkpoints
    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
    """
    arch = type(model).__name__

    # data parallel add a module attribute, strip it off if necessary
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    # Save checkpoint as "latest" 
    filename = os.path.join(checkpoint_dir, 'checkpoint-latest.pth')
    torch.save(state, filename)
    logger.info("Saving checkpoint: {} ...".format(filename))

    # Save epoch checkpoint periodically
    epoch_save_mod = 100
    if epoch % epoch_save_mod == 0:
        filename = os.path.join(checkpoint_dir, 'checkpoint-epoch{:03d}.pth'
                                .format(epoch))
        torch.save(state, filename)
        logger.info("Saving checkpoint: {} ...".format(filename))

    # Save best also if requested
    if best_flag:
        filename = os.path.join(checkpoint_dir, 'best-model.pth')
        torch.save(state, filename)
        logger.info("Saving best model: {} ...".format(filename))

def resume_checkpoint(resume_path, model, optimizer, scheduler, device):
    """
    Resume from saved checkpoints
    :param resume_path: Checkpoint path to be resumed
    """
    logger.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, start_epoch))
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return start_epoch, model, optimizer, scheduler

def load_model(model_path, model, device):
    """
    Load model from path
    """
    logger.info("Loading model: {} ...".format(model_path))
    checkpoint = torch.load(model_path, map_location=device)
    load_keys = []
    try:
        for name, value in model.plda.named_parameters():
            load_keys.append('plda.'+name)
        for name, value in model.embed.named_parameters():
            load_keys.append('embed.'+name)
        for name, value in model.embed.named_buffers():
            load_keys.append('embed.'+name)
    except:
        load_keys = checkpoint['state_dict'].keys()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in load_keys}

    if hasattr(model, 'module'):
        model.module.load_state_dict(pretrained_dict, strict=False)
    else:
        model.load_state_dict(pretrained_dict, strict=False)
    logger.info("Model '{}' loaded".format(model_path))
    logger.info(" Successfully loaded keys: '{}'".format(load_keys))
    mod_keys = model.state_dict().keys()
    load_keys = pretrained_dict.keys()
    key_list = []
    for k in load_keys:
        if k not in mod_keys:
            key_list.append(k)
    if key_list:
        logger.info(" Ignoring loaded keys: '{}'".format(key_list))
    key_list = []
    for k in mod_keys:
        if k not in load_keys:
            key_list.append(k)
    if key_list:
        logger.info(" Didn't find loaded keys: '{}'".format(key_list))

    return model

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def bn_momentum_adjust(model, optimizer):
    """
    Adjust batch_norm layer momemtum
    """
    lr_thresh_list = [7e-4, 7e-5, 0.0]
    bn_mom_list = [0.1, 0.01, 0.001]
    # Get stepsize
    for i, param_group in enumerate(optimizer.param_groups):
        lr = float(param_group['lr'])
        logger.info("Stepsize discovered as %f" % (lr))
        break

    # Find desired momentum
    bn_mom = bn_mom_list[0]
    for i in range(len(lr_thresh_list)):
        if lr > lr_thresh_list[i]:
            bn_mom = bn_mom_list[i]
            break

    # Check if it's too big and reset
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            if m.momentum > bn_mom+(1e-8):
                logger.info("Adjusting bn momentum %f to %f" % (m.momentum,bn_mom))
                m.momentum = bn_mom
            else:
                break

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class NoamLR(torch.optim.lr_scheduler._LRScheduler): 
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number.
    700**-0.5 / 100**-0.5 = 7**-0.5 = 1/128
    
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]

class linear_up_downLR(torch.optim.lr_scheduler._LRScheduler): 
    """
    Parameters
    ----------
    N0: ``int``, required.
        The number of steps to linearly increase the learning rate.
    N: ``int``, required.
        The total number of steps.
    """
    def __init__(self, optimizer, N0, N, min_scale=(2e-3)):
        self.N0 = N0
        self.N = N
        self.min_scale = min_scale
        super().__init__(optimizer)

    def get_lr(self):
        n = self.last_epoch+1
        if n <= self.N0:
            scale = n / self.N0
        else:
            scale = (self.N-n)/(self.N-self.N0)
        scale = min(max(scale, self.min_scale),1.0)
        return [base_lr * scale for base_lr in self.base_lrs]
