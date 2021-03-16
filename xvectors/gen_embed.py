#!/usr/bin/env python
import torch
import numpy as np
import logging

from xvectors.utils import load_model


def load_embed_model(model_file, device='cpu'):

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    model = load_model(model_file, device)
    # get model
    model.eval()
    return model


def gen_embed(data, model):
    # TODO: automate sizing so that we can remove the flags ...

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
