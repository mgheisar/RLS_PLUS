import torch
from CPFLOW_lib.icnn import (ICNN, ICNN2, ICNN3, ResICNN2, DenseICNN2)
import random
import numpy


def batch_iter(X, batch_size=64, shuffle=False):
    """
    X: feature tensor (shape: num_instances x num_features)
    """
    if shuffle:
        idxs = torch.randperm(X.shape[0])
    else:
        idxs = torch.arange(X.shape[0])
    if X.is_cuda:
        idxs = idxs
    for batch_idxs in idxs.split(batch_size):
        yield X[batch_idxs]


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


def seed_prng(seed, cuda=False):
    random.seed(seed)
    numpy.random.seed(random.randint(1, 100000))
    torch.random.manual_seed(random.randint(1, 100000))
    if cuda is True:
        torch.cuda.manual_seed_all(random.randint(1, 100000))