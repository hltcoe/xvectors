from __future__ import print_function
from collections import OrderedDict
import logging
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from utils import accuracy

logger = logging.getLogger(__name__)


class X3(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, num_classes, expansion_rate=3, drop_p=0.0, bn_affine=False):

        super(X3, self).__init__()

        layers = []
        # conv blocks                                                                                                                                                                                                                                                                                         
        layers.extend(self.conv_block(1, input_dim, layer_dim, mid_dim, 5, 1, 0, bn_affine))
        layers.extend(self.conv_block(2, mid_dim, layer_dim, mid_dim, 3, 2, 0, bn_affine))
        layers.extend(self.conv_block(3, mid_dim, layer_dim, mid_dim, 3, 3, 0, bn_affine))
        layers.extend(self.conv_block(4, mid_dim, layer_dim, layer_dim, 3, 4, 0, bn_affine))

        # expansion layer                                                                                                                                                                                                                                                                                     
        layers.extend([('expand_linear', nn.Conv1d(layer_dim, layer_dim*expansion_rate, kernel_size=1)),
                       ('expand_relu', nn.LeakyReLU(inplace=True)),
                       ('expand_bn', nn.BatchNorm1d(layer_dim*expansion_rate, affine=False))])

        # Dropout pre-pooling                                                                                                                                                                                                                                                                                 
        if drop_p > 0.0:
            layers.extend([('drop_pre_pool', nn.Dropout2d(p=drop_p, inplace=True))])

        self.prepooling_layers = nn.Sequential(OrderedDict(layers))

        # pooling defined below                                                                                                                                                                                                                                                                               

        # embedding                                                                                                                                                                                                                                                                                           
        self.embedding = nn.Linear(layer_dim*expansion_rate*2, embedding_dim)

        out_layers = []
        out_layers.extend([('embedd_relu', nn.LeakyReLU(inplace=True)),
                           ('embedd_bn', nn.BatchNorm1d(embedding_dim, affine=bn_affine)),
                           ('out_linear', nn.Linear(embedding_dim, num_classes))])

        self.output = nn.Sequential(OrderedDict(out_layers))

        self.init_weight()

    def conv_block(self, index, in_channels, mid_channels, out_channels, kernel_size, dilation, padding, bn_affine=False):
         return [('conv%d' % index, nn.Conv1d(in_channels, mid_channels, kernel_size, dilation=dilation, padding=padding)),
                 ('relu%d' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%d' % index, nn.BatchNorm1d(mid_channels, affine=bn_affine)),
                 ('linear%d' % index, nn.Conv1d(mid_channels, out_channels, kernel_size=1)),
                 ('relu%da' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%da' % index, nn.BatchNorm1d(out_channels, affine=bn_affine))]

    def init_weight(self):
        """                                                                                                                                                                                                                                                                                                   
        Initialize weight with sensible defaults for the various layer types                                                                                                                                                                                                                                  
        :return:                                                                                                                                                                                                                                                                                              
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                logger.info("Initializing %s with xvaivier normal" % str(m))
                #nn.init.xavier_normal_(m.weight)                                                                                                                                                                                                                                                             
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU                                                                                                                                                                                                               
            
            if isinstance(m, nn.Linear):
                logger.info("Initializing %s with xvavier normal" % str(m))
                #nn.init.xavier_normal_(m.weight)                                                                                                                                                                                                                                                             
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU                                                                                                                                                                                                               

    def mean_std_pooling(self, x, eps=1e-9):
        m = torch.mean(x, dim=2)                                                                                                                                                                                                                                                        
        s = torch.sqrt(torch.mean((x - m.unsqueeze(2))**2, dim=2) + eps)
        x = torch.cat([m, s], dim=1)
        return x


    def extract_embedding(self, x):
        x = self.prepooling_layers(x)
        x = self.mean_std_pooling(x)
        x = self.embedding(x)
        return x

    def forward(self, x):
        # Compute embeddings                                                                                                                                                                                                                                                                                  
        x = self.extract_embedding(x)
        x = self.output(x)
        return x


