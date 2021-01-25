#!/usr/bin/env python
import torch
import numpy as np
import logging

from utils import load_model
from xvector_model import Xvector9s
from xvector_model_dgr import X3
from xvector_model_dgr2 import angular_ResNet

def load_embed_model(model_file, device='cpu'):

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    resnet_flag = 0
    dgr_flag = 0
    if 1:
        # length-norm Gaussian
        embed_relu_flag=False
        length_norm = True
    else:
        # xvec style
        embed_relu_flag=True
        length_norm = False
    num_classes=0
    if 0:
        input_dim=64
        layer_dim=1024
        embedding_dim=512
        num_classes=13129
        dgr_flag = 1
    elif 0:
        input_dim = 23
        layer_dim = 512
        embedding_dim = 256
        #embedding_dim = 13
        num_classes = 14
    elif 0:
        input_dim = 64
        layer_dim = 256
        embedding_dim = 256
        num_classes = 11816
    elif 1:
        input_dim = 64
        #layer_dim = 512
        layer_dim = 768
        embedding_dim = 128
        #embedding_dim = 64
        num_classes = 11816
    elif 1:
        input_dim = 80
        #layer_dim = 512
        layer_dim = 768
        embedding_dim = 128
        #embedding_dim = 256
        #num_classes = 5394
        #num_classes = 5694
        #num_classes = 5994
        num_classes = 6825
    elif 1:
        input_dim = 80
        #layer_dim = 768
        layer_dim = 1024
        embedding_dim = 256
        #num_classes = 5694
        num_classes = 5994
    else:
        input_dim = 64
        layer_dim = 512
        embedding_dim = 512

    if dgr_flag:
        if 0:
            model = X3(input_dim=80, layer_dim=512, mid_dim=512, embedding_dim=512, num_classes=7185, expansion_rate=3, drop_p=0.0, bn_affine=False)
            model = load_model(model_file, model, device)
        elif 1:
            scale=30.0
            margin=0.3
            inplanes=32
            model = angular_ResNet(input_dim=input_dim, embedding_dim=embedding_dim, inplanes=inplanes, num_classes=num_classes, margin=margin, scale=scale)
    else:
        model = Xvector9s(input_dim=input_dim,
                          layer_dim=layer_dim,
                          embedding_dim=embedding_dim,
                          num_classes=num_classes,
                          LL='None',
                          length_norm=length_norm,resnet_flag=resnet_flag,embed_relu_flag=embed_relu_flag).to(device)
        model = load_model(model_file, model, device)

    # get model
    model.eval()

    return model

def gen_embed(data, model):

    # generate embedding (x=raw, y=scaled length-norm, z=plda)
    embedding_only = True
    dgr_flag = 0
    with torch.no_grad():
        if dgr_flag:
            if 0:
                y = model.extract_embedding(torch.from_numpy(data).unsqueeze(0))
            else:
                y = model.extract_embedding(torch.from_numpy(data).unsqueeze(0).unsqueeze(1))
                #output = model.extract_embedding(torch.from_numpy(chunk_feats).unsqueeze(0).unsqueeze(1).to(device)).cpu().numpy().squeeze()
        else:
            x, y, z, output, w = model(torch.from_numpy(data).unsqueeze(0), embedding_only)
            #x, y, z, output, w = model(torch.from_numpy(data).unsqueeze(0))

    #return z.numpy().T
    return y.numpy().T

def get_plda(model_file):

    model = load_embed_model(model_file)
    Ulda = model.plda.Ulda.numpy()
    d_wc = model.plda.d_wc.numpy()
    d_ac = model.plda.d_ac.numpy()
    d_ac = np.maximum(d_ac,0.0)
    if 0:
        # keep normscale
        sc = model.plda.norm_scale.numpy()
        print("scale Ulda by %f" % sc)
        Ulda /= sc
    return Ulda, d_wc, d_ac
