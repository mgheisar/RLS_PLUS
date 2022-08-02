#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from contextlib import contextmanager
import gc
import math
import random
import os
import time
import warnings

import numpy
import numpy as np
import torch
from tqdm import tqdm

import lib.datasets as datasets
import lib.utils as utils
import lib.flows as flows
from lib.flows import (SequentialFlow, DeepConvexFlow, ActNorm)
from lib.icnn import (ICNN, ICNN2, ICNN3, ResICNN2, DenseICNN2)
from pingouin import multivariate_normality
import pandas as pd
import scipy.stats as stats

torch.cuda.set_device(0)
cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
################################################################################
#                               Helper Functions                               #
################################################################################

# noinspection PyShadowingNames
@contextmanager
def eval_ctx(flow, bruteforce=False, debug=False, no_grad=True):
    flow.eval()
    for f in flow.flows[1::2]:
        f.no_bruteforce = not bruteforce
    torch.autograd.set_detect_anomaly(debug)
    with torch.set_grad_enabled(mode=not no_grad):
        yield
    torch.autograd.set_detect_anomaly(False)
    for f in flow.flows[1::2]:
        f.no_bruteforce = True
    flow.train()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# noinspection PyShadowingNames
def seed_prng(seed, cuda=False):
    random.seed(seed)
    numpy.random.seed(random.randint(1, 100000))
    torch.random.manual_seed(random.randint(1, 100000))
    if cuda is True:
        torch.cuda.manual_seed_all(random.randint(1, 100000))


################################################################################
#                                    Parser                                    #
################################################################################


parser = argparse.ArgumentParser('Convex Potential Flow')
parser.add_argument(
    '--data', choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300'], type=str, default='wlatent'
)
parser.add_argument(
    '--arch', choices=['icnn', 'icnn2', 'icnn3', 'denseicnn2', 'resicnn2'], type=str, default='icnn2',
)
parser.add_argument(
    '--softplus_type', choices=['softplus', 'gaussian_softplus'], type=str, default='gaussian_softplus',
)
parser.add_argument(
    '--zero_softplus', type=eval, choices=[True, False], default=True,
)
parser.add_argument(
    '--symm_act_first', type=eval, choices=[True, False], default=False,
)
parser.add_argument(
    '--trainable_w0', type=eval, choices=[True, False], default=True,
)
parser.add_argument('--clip_grad', type=float, default=0)
parser.add_argument(
    '--preload_data', action='store_true', default=False,
    help="Preload entire dataset to GPU (if cuda).")

parser.add_argument('--save', type=str, default='experiments/cpfw_512_2')
parser.add_argument('--dimh', type=int, default=4096)  # 64:img
parser.add_argument('--nhidden', type=int, default=5)  # 4:img
parser.add_argument("--nblocks", type=int, default=10, help='Number of stacked CPFs.')  # 8-8-8 img

parser.add_argument("--nepochs", type=int, default=1000, help='Number of training epochs')
parser.add_argument('--early_stopping', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--val_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', help='Weight decay', type=float, default=1e-6)
parser.add_argument('--atol', type=float, default=1e-3)
parser.add_argument('--rtol', type=float, default=0.0)

parser.add_argument('--cuda', type=int, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data_root', type=str, default=None)
parser.add_argument('--resume', type=str, default=False)
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--fp64', action='store_true', default=False)
parser.add_argument('--val_freq', type=int, default=200)
parser.add_argument('--brute_val', action='store_true', default=False)
parser.add_argument('--log_freq', type=int, default=100)
parser.add_argument('--train_est_freq', type=int, default=None)
args = parser.parse_args()

args.val_batch_size = args.val_batch_size if args.val_batch_size else args.batch_size
args.test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size
args.train_est_freq = args.train_est_freq if args.train_est_freq else args.log_freq
a = datasets.root
args.data_root = args.data_root if args.data_root else datasets.root
log_path = os.path.join(args.save, 'logs')
# Logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=log_path)
logger.info(args)

if args.fp64:
    torch.set_default_dtype(torch.float64)


# noinspection PyPep8Naming
def batch_iter(X, batch_size=args.batch_size, shuffle=False):
    """
    X: feature tensor (shape: num_instances x num_features)
    """
    if shuffle:
        idxs = torch.randperm(X.shape[0])
    else:
        idxs = torch.arange(X.shape[0])
    if X.is_cuda:
        idxs = idxs.to(device)
    for batch_idxs in idxs.split(batch_size):
        yield X[batch_idxs]


def load_data(name, data_root):
    if name == 'bsds300':
        return datasets.BSDS300(data_root)

    elif name == 'power':
        return datasets.POWER(data_root)

    elif name == 'gas':
        return datasets.GAS(data_root)

    elif name == 'hepmass':
        return datasets.HEPMASS(data_root)

    elif name == 'miniboone':
        return datasets.MINIBOONE(data_root)

    elif name == 'wlatent':
        return datasets.WLATENT(data_root)

    else:
        raise ValueError('Unknown dataset.')


def load_arch(name):
    if name == 'icnn':
        return ICNN
    elif name == 'icnn2':
        return ICNN2
    elif name == 'icnn3':
        return ICNN3
    elif name == 'denseicnn2':
        return DenseICNN2
    elif name == 'resicnn2':
        return ResICNN2
    else:
        raise ValueError('Unknown input convex architecture.')


ndecs = 0
def update_lr(optimizer, n_vals_without_improvement):
    global ndecs
    if ndecs == 0 and n_vals_without_improvement > args.early_stopping // 3:
        base_lr = args.lr / 10
        ndecs = 1
    elif ndecs == 1 and n_vals_without_improvement > args.early_stopping // 3 * 2:
        base_lr = args.lr / 100
        ndecs = 2
    else:
        base_lr = args.lr / 10 ** ndecs
    for param_group in optimizer.param_groups:
        param_group["lr"] = base_lr
    return base_lr


# noinspection PyPep8Naming,PyShadowingNames
def train(model, trainD, evalD, checkpt=None):
    global ndecs
    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                             betas=(0.9, 0.99), weight_decay=args.wd)
    #  sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.nepochs * trainD.N)
    if checkpt is not None:
        optim.load_state_dict(checkpt['optim'])
        ndecs = checkpt['ndecs']

    batch_time = utils.RunningAverageMeter(0.98)
    cg_meter = utils.RunningAverageMeter(0.98)
    gnorm_meter = utils.RunningAverageMeter(0.98)
    train_est_meter = utils.RunningAverageMeter(0.98 ** args.train_est_freq)

    best_log = - float('inf')
    best_stat = 0.001
    best_stat_hz = 0.001
    itr = 0 if checkpt is None else checkpt['iters']
    n_vals_without_improvement = 0
    model.train()
    while True:
        if itr >= args.nepochs * math.ceil(trainD.N / args.batch_size):
            break
        if 0 < args.early_stopping < n_vals_without_improvement:
            break
        for x in batch_iter(trainD.x, shuffle=True):
            if 0 < args.early_stopping < n_vals_without_improvement:
                break
            end = time.time()
            optim.zero_grad()

            x = cvt(x)
            train_est = [0] if itr % args.train_est_freq == 0 else None
            loss = - model.logp(x, extra=train_est).mean()
            if train_est is not None:
                train_est = train_est[0].mean().detach().item()

            if loss != loss:
                raise ValueError('NaN encountered @ training logp!')

            loss.backward()

            if args.clip_grad == 0:
                parameters = [p for p in model.parameters() if p.grad is not None]
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in parameters]), 2.0)
            else:
                grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.clip_grad)

            optim.step()
            #  sch.step()

            gnorm_meter.update(float(grad_norm))
            cg_meter.update(sum(flows.CG_ITERS_TRACER))
            flows.CG_ITERS_TRACER.clear()
            batch_time.update(time.time() - end)
            if train_est is not None:
                train_est_meter.update(train_est)

            del loss
            gc.collect()
            torch.clear_autocast_cache()

            if itr % args.log_freq == 0:
                log_message = (
                    'Iter {:06d} | Epoch {:.2f} | Time {batch_time.val:.3f} | '
                    'GradNorm {gnorm_meter.avg:.2f} | CG iters {cg_meter.val} ({cg_meter.avg:.2f}) | '
                    'Train logp {train_logp.val:.6f} ({train_logp.avg:.6f})'.format(
                        itr,
                        float(itr) / (trainD.N / float(args.batch_size)),
                        batch_time=batch_time, gnorm_meter=gnorm_meter, cg_meter=cg_meter,
                        train_logp=train_est_meter
                    )
                )
                logger.info(log_message)

            # Validation loop.
            if itr % args.val_freq == 0:
                with eval_ctx(model, bruteforce=args.brute_val):
                    val_logp = utils.AverageMeter()
                    z_samples = []
                    with tqdm(total=evalD.N) as pbar:  # evalD.N
                        # noinspection PyAssignmentToLoopOrWithParameter
                        for x in batch_iter(evalD.x, batch_size=args.val_batch_size):  # evalD.x
                            x = cvt(x)
                            val_logp.update(model.logp(x).mean().item(), x.size(0))
                            z = model.forward_transform(x)[0]
                            z_samples.append(z.cpu().data.numpy())
                            pbar.update(x.size(0))
                    z_samples = np.vstack(z_samples)
                    p = np.zeros((data.n_dims,))
                    HZResults = multivariate_normality(pd.DataFrame(z_samples), alpha=.05)
                    val_stat_hz = HZResults.pval
                    for ii in range(data.n_dims):
                        x1 = z_samples[:, ii]
                        stat, p[ii] = stats.shapiro(x1)
                    val_stat = np.min(p)
                    if val_logp.avg > best_log:
                        best_log = val_logp.avg
                        utils.makedirs(args.save)
                        torch.save({
                            'args': args,
                            'model': model.state_dict(),
                            'optim': optim.state_dict(),
                            'iters': itr + 1,
                            'ndecs': ndecs,
                        }, os.path.join(args.save, 'best_model.pth'))
                        n_vals_without_improvement = 0
                        if val_stat_hz > 0.05:
                            utils.makedirs(args.save)
                            torch.save({
                                'args': args,
                                'model': model.state_dict(),
                                'optim': optim.state_dict(),
                                'iters': itr + 1,
                                'ndecs': ndecs,
                            }, os.path.join(args.save, 'best_model_stat_hz_logp.pth'))
                        if val_stat > 0.05:
                            utils.makedirs(args.save)
                            torch.save({
                                'args': args,
                                'model': model.state_dict(),
                                'optim': optim.state_dict(),
                                'iters': itr + 1,
                                'ndecs': ndecs,
                            }, os.path.join(args.save, 'best_model_stat_logp.pth'))
                    else:
                        n_vals_without_improvement += 1
                        update_lr(optim, n_vals_without_improvement)
                        torch.save({
                            'args': args,
                            'model': model.state_dict(),
                            'optim': optim.state_dict(),
                            'iters': itr + 1,
                            'ndecs': ndecs,
                        }, os.path.join(args.save, 'most_recent.pth'))

                    if val_stat_hz > best_stat_hz:
                        best_stat_hz = val_stat_hz
                        utils.makedirs(args.save)
                        torch.save({
                            'args': args,
                            'model': model.state_dict(),
                            'optim': optim.state_dict(),
                            'iters': itr + 1,
                            'ndecs': ndecs,
                        }, os.path.join(args.save, 'best_model_stat_hz.pth'))
                    if val_stat > best_stat:
                        best_stat = val_stat
                        utils.makedirs(args.save)
                        torch.save({
                            'args': args,
                            'model': model.state_dict(),
                            'optim': optim.state_dict(),
                            'iters': itr + 1,
                            'ndecs': ndecs,
                        }, os.path.join(args.save, 'best_model_stat.pth'))
                    log_message = (
                        '[VAL] Iter {:06d} | Val stat {:.6f} | Val hz stat {:.6f} | Val logp {:.6f} |'
                        'NoImproveEpochs {:02d}/{:02d}'.format(
                            itr, val_stat, val_stat_hz, val_logp.avg, n_vals_without_improvement, args.early_stopping
                        )
                    )
                    logger.info(log_message)

            itr += 1

    logger.info('Training has finished, yielding the best model...')
    best_checkpt = torch.load(os.path.join(args.save, 'best_model.pth'))
    model.load_state_dict(best_checkpt['model'])
    return model


if __name__ == '__main__':

    ################################################################################
    #                               Resolve Settings                               #
    ################################################################################

    # Device
    # cuda = torch.cuda.is_available() and args.cuda is not None
    # device = torch.device("cuda:" + str(args.cuda) if cuda else "cpu")
    dtype = torch.float32 if not args.fp64 else torch.float64

    # noinspection PyShadowingNames
    def cvt(x):
        return x.to(device=device, dtype=dtype, memory_format=torch.contiguous_format)
    logger.info('Using GPU: {} of the {}'.format(args.cuda if cuda else None,
                                                 torch.cuda.device_count()))

    # PRNG
    seed_prng(args.seed, cuda=cuda)

    ################################################################################
    #                                 Load Dataset                                 #
    ################################################################################

    data = load_data(args.data, args.data_root)
    data.trn.x = torch.from_numpy(data.trn.x)
    if args.preload_data is True and not args.evaluate:
        data.trn.x = cvt(data.trn.x)
    data.val.x = torch.from_numpy(data.val.x)
    if args.preload_data is True:
        data.val.x = cvt(data.val.x)
    data.tst.x = torch.from_numpy(data.tst.x)
    if args.preload_data is True:
        data.tst.x = cvt(data.tst.x)

    ################################################################################
    #                                Define Models                                 #
    ################################################################################

    Arch = load_arch(args.arch)
    icnns = [Arch(data.n_dims, args.dimh, args.nhidden,
                  softplus_type=args.softplus_type,
                  zero_softplus=args.zero_softplus,
                  symm_act_first=args.symm_act_first) for _ in range(args.nblocks)]
    layers = [None] * (2 * args.nblocks + 1)
    layers[0::2] = [ActNorm(data.n_dims) for _ in range(args.nblocks + 1)]
    layers[1::2] = [DeepConvexFlow(icnn, data.n_dims, unbiased=False,
                                   atol=args.atol, rtol=args.rtol,
                                   trainable_w0=args.trainable_w0) for _, icnn in zip(range(args.nblocks), icnns)]

    flow = SequentialFlow(layers)
    flow = flow.to(device=device, dtype=dtype)
    checkpt = None
    try:
        if args.resume:
            # deal with data-dependent initialization like actnorm.
            with torch.no_grad():
                x = torch.rand(8, data.n_dims).to(device)
                flow.forward_transform(x)

            checkpt = torch.load(os.path.join(args.save, 'most_recent.pth'))
            logger.info("Resuming from checkpoint @ %s", args.resume)
            flow.load_state_dict(checkpt['model'])
    except FileNotFoundError:
        warnings.warn("Resume file provided, but not found... starting from scratch: {}".format(
            args.resume))

    # logger.info(flow)
    logger.info("Number of trainable parameters:{}".format(count_parameters(flow)))

    ################################################################################
    #                                   Training                                   #
    ################################################################################
    if not args.evaluate:
        flow = train(flow, data.trn, data.val, checkpt)

    ################################################################################
    #                                   Testing                                    #
    ################################################################################

    logger.info('Evaluating model on test set.')
    with eval_ctx(flow, bruteforce=True):
        test_logp = utils.AverageMeter()
        with tqdm(total=data.tst.N) as pbar:
            for itr, x in enumerate(batch_iter(data.tst.x, batch_size=args.test_batch_size)):
                x = cvt(x)
                test_logp.update(flow.logp(x).mean().item(), x.size(0))
                pbar.update(x.size(0))
        log_message = '[TEST] Iter {:06d} | Test logp {:.6f}'.format(itr, test_logp.avg)
        logger.info(log_message)
