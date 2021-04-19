#!/usr/bin/env python

import argparse
import os
import torch
from shutil import copyfile
import logging


logger = logging.getLogger(__name__)


def add_info_to_model(model_fp, model_constructor_args):
    # make backup of model
    copyfile(model_fp, model_fp+'.backup')
    # read model in
    model = torch.load(model_fp, map_location=torch.device('cpu'))
    if 'model_constructor_args' in model:
        logger.warning('Warning! Overwriting existing model_constructor_args!!')
    # add necessary information
    model['model_constructor_args'] = model_constructor_args
    # save new model
    torch.save(model, model_fp)     


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default=None)
    parser.add_argument('--input-dim', type=int, required=True)
    parser.add_argument('--layer-dim', type=int, required=True)
    parser.add_argument('--embedding-dim', type=int, required=True)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--LL', type=str, choices=['linear', 'xvec', 'Gauss', 'Gauss_discr', 'None'], required=True)
    parser.add_argument('--N0', type=int, required=True)
    parser.add_argument('--fixed-N', action='store_true', default=False)
    parser.add_argument('--r', type=float, required=True)
    parser.add_argument('--enroll-type', type=str, required=True)  # can be 'Bayes', 'ML', or something else based on code ...
    parser.add_argument('--length-norm', action='store_true', default=False)
    parser.add_argument('--resnet-flag', action='store_true', default=False)
    parser.add_argument('--embed-relu', action='store_true', default=False)

    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise ValueError('--model argument must point to a file!')

    model_constructor_args = {
        'input_dim': args.input_dim,
        'layer_dim': args.layer_dim,
        'embedding_dim': args.embedding_dim,
        'num_classes': args.num_classes,
        'LL': args.LL,
        'N0': args.N0,
        'fixed_N': args.fixed_N,
        'r': args.r,
        'enroll_type': args.enroll_type,
        'length_norm': args.length_norm,
        'resnet_flag': args.resnet_flag,
        'embed_relu_flag': args.embed_relu
    }

    add_info_to_model(args.model, model_constructor_args)
 
