from __future__ import print_function
from collections import OrderedDict
import logging
import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from xvectors.utils import accuracy
"""
from sync_batchnorm import SynchronizedBatchNorm1d
import torchvision.models as torch_models

from effnet_utils import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
)
"""


logger = logging.getLogger(__name__)


#### LDE layer
class LDE(nn.Module):
    def __init__(self, input_dim, clusters, eps=1e-9):
        super(LDE, self).__init__()
        # dimensions and number of clusters
        self.input_dim = input_dim
        self.clusters = clusters
        self.eps = eps

        # Cluster means and bias   
        self.mean = nn.Parameter(torch.Tensor(clusters, input_dim))
        self.bias = nn.Parameter(torch.zeros(clusters))
        self.scale = nn.Parameter(torch.ones(clusters))

        # Init to means
        self.mean.data.uniform_(-0.1, 0.1)
                

    def forward(self, x):
        
        x_c = (x - self.mean[:,None,:,None]).permute(1,0,2,3)
        
        w_ct = F.softmax((-0.5 * self.scale[:,None]) * x_c.pow(2).sum(dim=2) + self.bias[:,None] , dim=1)
        
        #tot_w = w_ct.sum(2) + self.eps
        
        #w_ct = w_ct / tot_w[:,:,None]            
        
        return torch.matmul(x_c, w_ct[:,:,:,None]).reshape(-1, self.clusters*self.input_dim)


#### Transformer

class Transformer_arc_X3(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, num_classes, drop_p=0.1, bn_affine=False, margin=0.1, scale=1.0, easy_margin=False, nhead=4):

        super(Transformer_arc_X3, self).__init__()

        layers = []
        # conv blocks                                                                                                                                                                                                                                                                                                          
        layers.extend(self.conv_block(1, input_dim, layer_dim, layer_dim, 5, 1, 0, bn_affine))        
        self.conv_layer = nn.Sequential(OrderedDict(layers))
        
        
        layers = []
        layers.extend([('Trans1',nn.TransformerEncoderLayer(d_model=layer_dim, nhead=nhead, dim_feedforward=mid_dim, dropout=drop_p)),
                       ('Trans2',nn.TransformerEncoderLayer(d_model=layer_dim, nhead=nhead, dim_feedforward=mid_dim, dropout=drop_p)),
                       ('Trans3',nn.TransformerEncoderLayer(d_model=layer_dim, nhead=nhead, dim_feedforward=mid_dim, dropout=drop_p))])
        
  
        self.transformer_layers = nn.Sequential(OrderedDict(layers))

        # pooling defined below                                                                                                                                                                                                                                                                                                

        # embedding                                                                                                                                                                                                                                                                                                            
        #self.embedding = nn.Linear(layer_dim*expansion_rate*2, embedding_dim)
        self.embedding = nn.Linear(layer_dim*2, embedding_dim)                                                                                                                                                                                                                                   
        self.base_margin = margin

        self.arc_logit = ArcMarginModel(num_classes, embedding_dim, easy_margin, margin, 1.0)

        self.out_scale = nn.Parameter(torch.log(torch.tensor(scale)))

        self.init_weight()

    def conv_block(self, index, in_channels, mid_channels, out_channels, kernel_size, dilation, padding, bn_affine=False):
         return [('conv%d' % index, nn.Conv1d(in_channels, mid_channels, kernel_size, dilation=dilation, padding=padding)),
                 ('relu%d' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%d' % index, nn.BatchNorm1d(mid_channels, affine=bn_affine)),
                 ('linear%d' % index, nn.Conv1d(mid_channels, out_channels, kernel_size=1)),
                 ('relu%da' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%da' % index, nn.BatchNorm1d(out_channels, affine=bn_affine))]

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                logger.info("Initializing %s with xaivier normal" % str(m))
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU                                                                                                                                                                                                                                
            if isinstance(m, nn.Linear):
                logger.info("Initializing %s with xavier normal" % str(m))
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU                                                                                                                                                                                                                                
    def extract_embedding(self, x):
        x = self.conv_layer(x)
        x = self.transformer_layers(x.transpose_(1,2))
        x = self.mean_std_pooling(x.transpose_(1,2))
        x = self.embedding(x)
        return x


    def mean_std_pooling(self, x, eps=1e-9):
        m = torch.mean(x, dim=2)                                                                                                                                                                                
        s = torch.sqrt(torch.mean((x - m.unsqueeze(2))**2, dim=2) + eps)
        x = torch.cat([m, s], dim=1)
        return x


    def forward(self, x, labels, epoch):
        # Compute embeddings                                                                                                                                                                                                                                                                                                                                                                                                    
        x = self.extract_embedding(x)
        if epoch < 2:
            self.arc_logit.set_margin(0.0)
        elif epoch < 4:
            self.arc_logit.set_margin(self.base_margin / 2.0)
        else:
            self.arc_logit.set_margin(self.base_margin)


        x = self.arc_logit(x, labels)
        this_scale = torch.exp(self.out_scale)
        x = this_scale * x
        return x


##### X10

class X10(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, num_classes, expansion_rate=3, drop_p=0.0, bn_affine=False):

        super(X10, self).__init__()

        layers = []
        # conv blocks                                                                                                                                                                                                                                                                                         
        layers.extend(self.conv_block(1, input_dim, layer_dim, mid_dim, 5, 1, 0, bn_affine))
        layers.extend(self.conv_block(2, mid_dim, layer_dim, mid_dim, 3, 2, 0, bn_affine))
        layers.extend(self.conv_block(3, mid_dim, layer_dim, mid_dim, 3, 3, 0, bn_affine))
        layers.extend(self.conv_block(4, mid_dim, layer_dim, mid_dim, 3, 4, 0, bn_affine))
        layers.extend(self.conv_block(5, mid_dim, layer_dim, layer_dim, 3, 5, 0, bn_affine))

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




##### X9

class X9(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, num_classes, expansion_rate=3, drop_p=0.0, bn_affine=False):

        super(X9, self).__init__()

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
        self.embedding = nn.Linear(layer_dim*expansion_rate, embedding_dim)

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
        #x = torch.cat([m, s], dim=1)
        x = m / s
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





##### X8

class X8(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, num_classes, expansion_rate=3, drop_p=0.0, bn_affine=False):

        super(X8, self).__init__()

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
        self.embedding = nn.Linear(layer_dim*expansion_rate, embedding_dim)

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

    def mean_pooling(self, x, eps=1e-9):
        m = torch.mean(x, dim=2)                                                                                                                                                                                                                                                                
        return m


    def extract_embedding(self, x):
        x = self.prepooling_layers(x)
        x = self.mean_pooling(x)
        x = self.embedding(x)
        return x

    def forward(self, x):
        # Compute embeddings                                                                                                                                                                                                                                                                                  
        x = self.extract_embedding(x)
        x = self.output(x)
        return x




##### X5
class X5(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, num_classes, num_clusters=64, h_dim=64, bn_affine=False, bn_pre_pool=False):

        super(X5, self).__init__()

        layers = []
        # conv blocks                                                                                                                                                                                                                                                                                         
        layers.extend(self.conv_block(1, input_dim, layer_dim, mid_dim, 5, 1, 0, bn_affine))
        layers.extend(self.conv_block(2, mid_dim, layer_dim, mid_dim, 3, 2, 0, bn_affine))
        layers.extend(self.conv_block(3, mid_dim, layer_dim, mid_dim, 3, 3, 0, bn_affine))
        layers.extend(self.conv_block(4, mid_dim, layer_dim, layer_dim, 3, 4, 0, bn_affine))

        # expansion layer                                                                                                                                                                                                                                                                                     
        layers.extend([('expand_linear', nn.Conv1d(layer_dim, h_dim, kernel_size=1)),
                       ('expand_relu', nn.LeakyReLU(inplace=True))])

        if bn_pre_pool:
            layers.extend([('bn_pre_pool', nn.BatchNorm1d(h_dim, affine=False))])

        self.prepooling_layers = nn.Sequential(OrderedDict(layers))

        # pooling LDE
        self.lde = LDE(h_dim, num_clusters)

        # embedding                                                                                                                                                                                                                                                                                           
        self.embedding = nn.Linear(h_dim*num_clusters, embedding_dim)

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

    def extract_embedding(self, x):
        x = self.prepooling_layers(x)
        x = self.lde(x)
        x = self.embedding(x)
        return x

    def forward(self, x):
        # Compute embeddings                                                                                                                                                                                                                                                                                  
        x = self.extract_embedding(x)
        x = self.output(x)
        return x



##### X6
class X6(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, num_classes, expansion_rate=3, drop_p=0.0, bn_affine=False):

        super(X6, self).__init__()

        self.embedding_dim = embedding_dim

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
        out_layers.extend([('out_linear', nn.Linear(embedding_dim, num_classes))])

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
        x = F.normalize(x, p=2, dim=1) * torch.sqrt(torch.tensor(self.embedding_dim, dtype=torch.float))
        x = self.output(x)
        return x



##### P1
class P1(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, expansion_rate=3, drop_p=0.0, bn_affine=False, init_scale=1.0):

        super(P1, self).__init__()

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

        self.output = nn.BatchNorm1d(embedding_dim, affine=bn_affine)

        # Scale by learned value
        self.out_scale = nn.Parameter(torch.tensor(init_scale))

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
        x = self.output(x)
        return x

    def forward(self, x):
        # Compute embeddings                                                                                                                                                                                                                                                                                  
        x = self.extract_embedding(x)
        #x = self.output(x)
        x = self.out_scale * F.normalize(x)
        #print('Scale:', self.out_scale)
        return x



##### X3

class X3(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, num_classes, expansion_rate=3, drop_p=0.0, bn_affine=False, stride=1):

        super(X3, self).__init__()

        layers = []
        # conv blocks                                                                                                                                                                                                                                                                                         
        layers.extend(self.conv_block(1, input_dim, layer_dim, mid_dim, 5, 1, 0, bn_affine, stride))
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

    def conv_block(self, index, in_channels, mid_channels, out_channels, kernel_size, dilation, padding, bn_affine=False, stride=1):
         return [('conv%d' % index, nn.Conv1d(in_channels, mid_channels, kernel_size, dilation=dilation, padding=padding, stride=stride)),
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
                logger.info("Initializing %s with kaiming normal" % str(m))
                #nn.init.xavier_normal_(m.weight)                                                                                                                                                                                                                                                             
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU                                                                                                                                                                                                               
            
            if isinstance(m, nn.Linear):
                logger.info("Initializing %s with kaiming normal" % str(m))
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


#### ARC-X3


class arc_X3(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, num_classes, expansion_rate=3, drop_p=0.0, bn_affine=False, margin=0.1, scale=50.0, easy_margin=False):

        super(arc_X3, self).__init__()

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

        #out_layers = []
        #out_layers.extend([('embedd_relu', nn.LeakyReLU(inplace=True)),
        #                   ('embedd_bn', nn.BatchNorm1d(embedding_dim, affine=bn_affine)),
        #                   ('out_linear', nn.Linear(embedding_dim, num_classes))])


        self.arc_logit = ArcMarginModel(num_classes, embedding_dim, easy_margin, margin, scale)
        
        self.base_margin = margin
        #self.output = nn.Sequential(OrderedDict(out_layers))

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

#    def forward(self, x, labels):
#        # Compute embeddings                                                                                                                                                                                   #                                                                                               
#        x = self.extract_embedding(x)
#        x = self.arc_logit(x, labels)
#        return x


    def forward(self, x, labels, epoch):
        # Compute embeddings                                                                                                                                                                                                                                                                                  
        x = self.extract_embedding(x)
        if epoch < 2:
            self.arc_logit.set_margin(0.0)
        #elif epoch < 4:
        #    self.arc_logit.set_margin(self.base_margin / 2.0)
        else:
            self.arc_logit.set_margin(self.base_margin)
            
        x = self.arc_logit(x, labels)

        return x






#### Scaled arc_X3

class scaled_arc_X3(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, num_classes, expansion_rate=3, drop_p=0.0, bn_affine=False, margin=0.1, scale=1.0, easy_margin=False):

        super(scaled_arc_X3, self).__init__()

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

        #out_layers = []
        #out_layers.extend([('embedd_relu', nn.LeakyReLU(inplace=True)),
        #                   ('embedd_bn', nn.BatchNorm1d(embedding_dim, affine=bn_affine)),
        #                   ('out_linear', nn.Linear(embedding_dim, num_classes))])


        self.arc_logit = ArcMarginModel(num_classes, embedding_dim, easy_margin, margin, 1.0)

        self.out_scale = nn.Parameter(torch.log(torch.tensor(scale)))

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

    def forward(self, x, labels):
        # Compute embeddings                                                                                                                                                                                                                                                                                  
        x = self.extract_embedding(x)
        x = self.arc_logit(x, labels)
        x = torch.exp(self.out_scale) * x
        return x



##### inc_arc_X3


class inc_arc_X3(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, num_classes, expansion_rate=1, drop_p=0.0, bn_affine=False, margin=0.1, scale=1.0, easy_margin=False):

        super(inc_arc_X3, self).__init__()

        layers = []
        # conv blocks                                                                                                                                                                                                                                                                                         
        layers.extend(self.conv_block(1, input_dim, layer_dim // 8, mid_dim // 8, 5, 1, 0, bn_affine))
        layers.extend(self.conv_block(2, mid_dim // 8, layer_dim // 4, mid_dim // 4, 3, 2, 0, bn_affine))
        layers.extend(self.conv_block(3, mid_dim // 4, layer_dim // 2, mid_dim // 2, 3, 3, 0, bn_affine))
        layers.extend(self.conv_block(4, mid_dim // 2, layer_dim, layer_dim, 3, 4, 0, bn_affine))

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

        #out_layers = []
        #out_layers.extend([('embedd_relu', nn.LeakyReLU(inplace=True)),
        #                   ('embedd_bn', nn.BatchNorm1d(embedding_dim, affine=bn_affine)),
        #                   ('out_linear', nn.Linear(embedding_dim, num_classes))])


        self.base_margin = margin

        self.arc_logit = ArcMarginModel(num_classes, embedding_dim, easy_margin, margin, 1.0)

        self.out_scale = nn.Parameter(torch.log(torch.tensor(scale)))

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

    def forward(self, x, labels, epoch):
        # Compute embeddings                                                                                                                                                                                                                                                                                  
        x = self.extract_embedding(x)
        if epoch < 2:
            self.arc_logit.set_margin(0.0)
        elif epoch < 4:
            self.arc_logit.set_margin(self.base_margin / 2.0)
        else:
            self.arc_logit.set_margin(self.base_margin)

        #print(self.arc_logit.m)
            
        x = self.arc_logit(x, labels)
        #x = torch.exp(self.out_scale) * x
        #this_scale = torch.exp(self.out_scale.clamp_min(2.0))
        this_scale = torch.exp(self.out_scale)
        #print(self.out_scale.item())
        x = this_scale * x
        return x



#### Scaled arc_X3

class scaled_margin_arc_X3(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, num_classes, expansion_rate=3, drop_p=0.0, bn_affine=False, margin=0.1, scale=1.0, easy_margin=False):

        super(scaled_margin_arc_X3, self).__init__()

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

        #out_layers = []
        #out_layers.extend([('embedd_relu', nn.LeakyReLU(inplace=True)),
        #                   ('embedd_bn', nn.BatchNorm1d(embedding_dim, affine=bn_affine)),
        #                   ('out_linear', nn.Linear(embedding_dim, num_classes))])


        self.base_margin = margin

        self.arc_logit = ArcMarginModel(num_classes, embedding_dim, easy_margin, margin, 1.0)

        self.out_scale = nn.Parameter(torch.log(torch.tensor(scale)))

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

    def forward(self, x, labels, epoch):
        # Compute embeddings                                                                                                                                                                                                                                                                                  
        x = self.extract_embedding(x)
        if epoch < 2:
            self.arc_logit.set_margin(0.0)
        elif epoch < 4:
            self.arc_logit.set_margin(self.base_margin / 2.0)
        else:
            self.arc_logit.set_margin(self.base_margin)

        #print(self.arc_logit.m)
            
        x = self.arc_logit(x, labels)
        #x = torch.exp(self.out_scale) * x
        #this_scale = torch.exp(self.out_scale.clamp_min(2.0))
        this_scale = torch.exp(self.out_scale)
        #print(self.out_scale.item())
        x = this_scale * x
        return x



#### Resume 

class resume_scaled_margin_arc_X3(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, num_classes, expansion_rate=3, drop_p=0.0, bn_affine=False, margin=0.1, scale=1.0, easy_margin=False):

        super(resume_scaled_margin_arc_X3, self).__init__()

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

        #out_layers = []
        #out_layers.extend([('embedd_relu', nn.LeakyReLU(inplace=True)),
        #                   ('embedd_bn', nn.BatchNorm1d(embedding_dim, affine=bn_affine)),
        #                   ('out_linear', nn.Linear(embedding_dim, num_classes))])


        self.base_margin = margin

        self.arc_logit = ArcMarginModel(num_classes, embedding_dim, easy_margin, margin, scale)

        self.out_scale = nn.Parameter(torch.log(torch.tensor(scale)))

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

    def forward(self, x, labels, epoch):
        # Compute embeddings                                                                                                                                                                                                                                                                                  
        x = self.extract_embedding(x)
        if epoch < 2:
            self.arc_logit.set_margin(0.0)
        elif epoch < 4:
            self.arc_logit.set_margin(self.base_margin / 2.0)
        else:
            self.arc_logit.set_margin(self.base_margin)

        #print(self.arc_logit.m)
            
        x = self.arc_logit(x, labels)
        return x





#### Scaled BN arc_X3

class scaled_bn_arc_X3(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, num_classes, expansion_rate=3, drop_p=0.0, bn_affine=False, margin=0.1, scale=1.0, easy_margin=False):

        super(scaled_bn_arc_X3, self).__init__()

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

        self.output = nn.BatchNorm1d(embedding_dim, affine=bn_affine)

        #out_layers = []
        #out_layers.extend([('embedd_relu', nn.LeakyReLU(inplace=True)),
        #                   ('embedd_bn', nn.BatchNorm1d(embedding_dim, affine=bn_affine)),
        #                   ('out_linear', nn.Linear(embedding_dim, num_classes))])


        self.arc_logit = ArcMarginModel(num_classes, embedding_dim, easy_margin, margin, 1.0)

        self.out_scale = nn.Parameter(torch.log(torch.tensor(scale)))

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
        x = self.output(x)
        return x

    def forward(self, x, labels):
        # Compute embeddings                                                                                                                                                                                                                                                                                  
        x = self.extract_embedding(x)
        x = self.arc_logit(x, labels)
        x = torch.exp(self.out_scale.clamp_min(0.0)) * x
        return x




##### X4

class X4(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, num_classes, expansion_rate=3, drop_p=0.0, bn_affine=False):

        super(X4, self).__init__()

        layers = []
        # conv blocks                                                                                                                                                                                                                                                                                         
        layers.extend(self.conv_block(1, input_dim, layer_dim, mid_dim, 5, 1, 0, bn_affine))
        layers.extend(self.conv_block(2, mid_dim, layer_dim, mid_dim, 3, 2, 0, bn_affine))
        layers.extend(self.conv_block(3, mid_dim, layer_dim, mid_dim, 3, 3, 0, bn_affine))
        layers.extend(self.conv_block(4, mid_dim, layer_dim, layer_dim, 3, 4, 0, bn_affine))

        # expansion layer                                                                                                                                                                                                                                                                                     
        layers.extend([('expand_linear', nn.Conv1d(layer_dim, layer_dim*expansion_rate, kernel_size=1)),                       
                       ('expand_bn', nn.BatchNorm1d(layer_dim*expansion_rate, affine=bn_affine)),
                       ('expand_relu', nn.LeakyReLU(inplace=True))])

        # Dropout pre-pooling                                                                                                                                                                                                                                                                                 
        if drop_p > 0.0:
            layers.extend([('drop_pre_pool', nn.Dropout2d(p=drop_p, inplace=True))])

        self.prepooling_layers = nn.Sequential(OrderedDict(layers))

        # pooling defined below                                                                                                                                                                                                                                                                               

        # embedding                                                                                                                                                                                                                                                                                           
        self.embedding = nn.Linear(layer_dim*expansion_rate*2, embedding_dim)

        out_layers = []
        out_layers.extend([('embedd_bn', nn.BatchNorm1d(embedding_dim, affine=False)),
                           ('embedd_relu', nn.LeakyReLU(inplace=True)),                           
                           ('out_linear', nn.Linear(embedding_dim, num_classes))])

        self.output = nn.Sequential(OrderedDict(out_layers))

        self.init_weight()

    def conv_block(self, index, in_channels, mid_channels, out_channels, kernel_size, dilation, padding, bn_affine=False):
         return [('conv%d' % index, nn.Conv1d(in_channels, mid_channels, kernel_size, dilation=dilation, padding=padding)),
                 ('bn%d' % index, nn.BatchNorm1d(mid_channels, affine=bn_affine)),
                 ('relu%d' % index, nn.LeakyReLU(inplace=True)),                 
                 ('linear%d' % index, nn.Conv1d(mid_channels, out_channels, kernel_size=1)),
                 ('bn%da' % index, nn.BatchNorm1d(out_channels, affine=bn_affine)),
                 ('relu%da' % index, nn.LeakyReLU(inplace=True))]

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




###### 




class TransposeLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(TransposeLinear, self).__init__(in_features, out_features, True)

    def forward(self, input):
        return F.linear(input.transpose(2, 1), self.weight, self.bias).transpose(2, 1)


class scale_ArcMarginModel(nn.Module):
    def __init__(self, num_classes, emb_size, easy_margin, margin_m, margin_s=1.0):
        super(scale_ArcMarginModel, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.m = margin_m
        self.out_scale = nn.Parameter(torch.tensor(margin_s))

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        # Only apply margin in trainng mode                                                                                                                                                                                                                                                                                    
        if self.training:
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)                                                                                                                                                                                                                                                    
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            one_hot = torch.zeros(cosine.size(),device=label.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            output = cosine

        output *= self.out_scale
        return output


class ArcMarginModel(nn.Module):
    def __init__(self, num_classes, emb_size, easy_margin, margin_m, margin_s=1.0):
        super(ArcMarginModel, self).__init__()        

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.m = margin_m
        self.s = margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        # Only apply margin in trainng mode
        if self.training:
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            one_hot = torch.zeros(cosine.size(),device=label.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            output = cosine

        output *= self.s
        return output

    def set_margin(self, value):
        self.m = value
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m


class ArcUncertainty(nn.Module):
    def __init__(self, num_classes, emb_size, max_s):
        super(ArcUncertainty, self).__init__()        

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_size))
        nn.init.xavier_uniform_(self.weight)
        self.max_s=max_s

    def forward(self, input, scale):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = self.max_s * F.linear(x, W)
        output = cosine * scale
        return output


# Multi-head attention module
class MH_att(nn.Module):
    def __init__(self, in_dim, h_dim=128, n_heads=32, n_stages=0, bn_affine=False):
        super(MH_att, self).__init__()
        
        # Hidden feature to use for weighted average        
        layers = []        
        layers.extend([('linear_h', TransposeLinear(in_dim, h_dim)),
                       ('relu_h', nn.LeakyReLU(inplace=True)),
                       ('bn_h', nn.BatchNorm1d(h_dim, affine=bn_affine))])        
        self.h_layers = nn.Sequential(OrderedDict(layers))
        
        # Multi-head attention
        layers = []        
        for i in range(n_stages):
            layers.extend(self.att_block(i,h_dim))
        layers.extend([('linear_att_final', TransposeLinear(h_dim, n_heads)), ('softmax_att', nn.Softmax(dim=1))])    
        self.att_layers = nn.Sequential(OrderedDict(layers))
        
        self.bn_out = nn.BatchNorm1d(h_dim*n_heads, affine=bn_affine)        
        
    def att_block(self, index, h_dim):
        return [('linear_att_%d' % index, TransposeLinear(h_dim, h_dim)),
                ('relu_att_%d' % index, nn.LeakyReLU(inplace=True)),
                ('bn_att_%d' % index, nn.BatchNorm1d(h_dim, affine=bn_affine))]
        
    def forward(self, x):
        h = self.h_layers(x)
        w = self.att_layers(h).transpose(2, 1)
        out = torch.matmul(h,w).reshape(-1,h.shape[1]*w.shape[2])
        return self.bn_out(out)





### arc_unc
class Xvector9s_arc_unc(nn.Module):

    def __init__(self, input_dim, layer_dim, embedding_dim, num_classes, h_dim=8, n_stages=0, max_scale=40.0, bn_affine=False):

        super(Xvector9s_arc_unc, self).__init__()
        self.prepooling_frozen = False
        self.embedding_frozen = False

        layers = []
        # conv blocks
        layers.extend(self.conv_block(1, input_dim, layer_dim, 5, 1, 2, bn_affine))
        layers.extend(self.conv_block(2, layer_dim, layer_dim, 3, 2, 2, bn_affine))
        layers.extend(self.conv_block(3, layer_dim, layer_dim, 3, 3, 3, bn_affine))
        layers.extend(self.conv_block(4, layer_dim, layer_dim, 3, 4, 4, bn_affine))

        # expansion layer
        layers.extend([('expand_conv', TransposeLinear(layer_dim, layer_dim*3)),
                       ('expand_relu', nn.LeakyReLU(inplace=True)),
                       ('expand_bn', nn.BatchNorm1d(layer_dim*3, affine=bn_affine))])

        self.prepooling_layers = nn.Sequential(OrderedDict(layers))

        # pooling defined below

        # embedding
        self.embedding = nn.Linear(layer_dim*6, embedding_dim)

        # Uncertainty computation layers
        layers = []
        layers.extend(self.unc_block(0,layer_dim*6, h_dim, bn_affine))

        for i in range(n_stages):
            layers.extend(self.unc_block(i+1,h_dim, h_dim, bn_affine))

        layers.extend([('unc_scalar', nn.Linear(h_dim, 1)),
                       ('unc_sigm', nn.Sigmoid())])
        self.unc_layers = nn.Sequential(OrderedDict(layers))

        # Cosine score scaled by uncertainty
        self.arc_logit = ArcUncertainty(num_classes, embedding_dim, max_scale)
            
        self.init_weight()


    def unc_block(self, index, h_in, h_out, bn_affine):
        return [('linear_unc_%d' % index, nn.Linear(h_in, h_out)),
                ('relu_unc_%d' % index, nn.LeakyReLU(inplace=True)),
                ('bn_unc_%d' % index, nn.BatchNorm1d(h_out, affine=bn_affine))]

    def conv_block(self, index, in_channels, out_channels, kernel_size, dilation, padding, bn_affine):
         return [('conv%d' % index, nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)),
                 ('relu%d' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%d' % index, nn.BatchNorm1d(out_channels, affine=bn_affine)),
                 ('linear%d' % index, TransposeLinear(out_channels, out_channels)),
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
            if isinstance(m, GaussLinear):
                logger.info("Initializing %s with zeros" % str(m))
            if isinstance(m, nn.Linear):
                logger.info("Initializing %s with xvavier normal" % str(m))
                #nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU

    def mean_std_pooling(self, x, eps=1e-9):
        # mean
        m = torch.mean(x, dim=2)

        # std
        # NOTE: std has stability issues as autograd of std(0) is NaN
        # we can use var, but that is not in ONNX export, so
        # do it the long way
        #s = torch.sqrt(torch.var(x, dim=2) + eps)
        s = torch.sqrt(torch.mean((x - m.unsqueeze(2))**2, dim=2) + eps)

        x = torch.cat([m, s], dim=1)

        return x

    def extract_embedding(self, x):
        x = self.prepooling_layers(x)
        x = self.mean_std_pooling(x)
        s = self.unc_layers(x)
        x = self.embedding(x)
        return torch.cat([x, s], dim=1)

    def forward(self, x):
        # Compute embeddings 
        #x = self.extract_embedding(x)        
        x = self.prepooling_layers(x)
        x = self.mean_std_pooling(x)
        s = self.unc_layers(x)
        x = self.embedding(x)
        x = self.arc_logit(x, s)
        return x


    def freeze_prepooling(self):
        # set model to eval mode and turn off gradients                                                                                                                                                                                                                                                                        
        self.prepooling_frozen = True
        for param in self.prepooling_layers.parameters():
            param.requires_grad = False

    def freeze_embedding(self):

        # Freeze embedding and everything before it                                                                                                                                                                                                                                                                            
        self.embedding_frozen = True
        self.freeze_prepooling()
        for param in self.embedding.parameters():
            param.requires_grad = False

    def train_with_freeze(self):
        # Set training mode except for frozen layers                                                                                                                                                                                                                                                                          
        self.train()
        if self.prepooling_frozen:
            self.prepooling_layers.eval()
        if self.embedding_frozen:
            self.embedding.eval()




### acrmargin v2
class Xvector9s_arc_v2(nn.Module):

    def __init__(self, input_dim, layer_dim, embedding_dim, num_classes, margin=0.5, scale=64.0, easy_margin=False, bn_affine=False):

        super(Xvector9s_arc_v2, self).__init__()
        self.prepooling_frozen = False
        self.embedding_frozen = False

        layers = []
        # conv blocks
        layers.extend(self.conv_block(1, input_dim, layer_dim, 5, 1, 2, bn_affine))
        layers.extend(self.conv_block(2, layer_dim, layer_dim, 3, 2, 2, bn_affine))
        layers.extend(self.conv_block(3, layer_dim, layer_dim, 3, 3, 3, bn_affine))
        layers.extend(self.conv_block(4, layer_dim, layer_dim, 3, 4, 4, bn_affine))

        # expansion layer
        layers.extend([('expand_conv', TransposeLinear(layer_dim, layer_dim*3)),
                       ('expand_relu', nn.LeakyReLU(inplace=True)),
                       ('expand_bn', nn.BatchNorm1d(layer_dim*3, affine=bn_affine))])

        self.prepooling_layers = nn.Sequential(OrderedDict(layers))

        # pooling defined below

        # embedding
        self.embedding = nn.Linear(layer_dim*6, embedding_dim)
        
        self.arc_logit = ArcMarginModel(num_classes, embedding_dim, easy_margin, margin, scale)
            
        self.init_weight()

    def conv_block(self, index, in_channels, out_channels, kernel_size, dilation, padding, bn_affine):
         return [('conv%d' % index, nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)),
                 ('relu%d' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%d' % index, nn.BatchNorm1d(out_channels, affine=bn_affine)),
                 ('linear%d' % index, TransposeLinear(out_channels, out_channels)),
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
            if isinstance(m, GaussLinear):
                logger.info("Initializing %s with zeros" % str(m))
            if isinstance(m, nn.Linear):
                logger.info("Initializing %s with xvavier normal" % str(m))
                #nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU

    def mean_std_pooling(self, x, eps=1e-9):
        # mean
        N = x.shape[0]
        T = x.shape[2]
        m = torch.mean(x, dim=2)

        # std
        # NOTE: std has stability issues as autograd of std(0) is NaN
        # we can use var, but that is not in ONNX export, so
        # do it the long way
        #s = torch.sqrt(torch.var(x, dim=2) + eps)
        s = torch.sqrt(torch.mean((x - m.unsqueeze(2))**2, dim=2) + eps)

        x = torch.cat([m, s], dim=1)

        return x


    def extract_embedding(self, x):
        x = self.prepooling_layers(x)
        x = self.mean_std_pooling(x)
        x = self.embedding(x)
        return x

    def forward(self, x, labels):
        # Compute embeddings 
        x = self.extract_embedding(x)        
        x = self.arc_logit(x, labels)
        return x

    def freeze_prepooling(self):
        # set model to eval mode and turn off gradients
        self.prepooling_frozen = True
        for param in self.prepooling_layers.parameters():
            param.requires_grad = False

    def freeze_embedding(self):

        # Freeze embedding and everything before it
        self.embedding_frozen = True
        self.freeze_prepooling()
        for param in self.embedding.parameters():
            param.requires_grad = False

    def train_with_freeze(self):

        # Set training mode except for frozen layers
        self.train()
        if self.prepooling_frozen:
            self.prepooling_layers.eval()
        if self.embedding_frozen:
            self.embedding.eval()



#### arc v3 (no BN pre-pooling)
class Xvector9s_arc_v3(nn.Module):

    def __init__(self, input_dim, layer_dim, embedding_dim, num_classes, margin=0.5, scale=64.0, easy_margin=False, bn_affine=False):

        super(Xvector9s_arc_v3, self).__init__()
        self.prepooling_frozen = False
        self.embedding_frozen = False

        layers = []
        # conv blocks
        layers.extend(self.conv_block(1, input_dim, layer_dim, 5, 1, 2, bn_affine))
        layers.extend(self.conv_block(2, layer_dim, layer_dim, 3, 2, 2, bn_affine))
        layers.extend(self.conv_block(3, layer_dim, layer_dim, 3, 3, 3, bn_affine))
        layers.extend(self.conv_block(4, layer_dim, layer_dim, 3, 4, 4, bn_affine))

        # expansion layer
        layers.extend([('expand_conv', TransposeLinear(layer_dim, layer_dim*3)),
                       ('expand_bn', nn.BatchNorm1d(layer_dim*3, affine=bn_affine)),
                       ('expand_relu', nn.LeakyReLU(inplace=True))])

        self.prepooling_layers = nn.Sequential(OrderedDict(layers))

        # pooling defined below

        # embedding
        self.embedding = nn.Linear(layer_dim*6, embedding_dim)
        
        self.arc_logit = ArcMarginModel(num_classes, embedding_dim, easy_margin, margin, scale)
            
        self.init_weight()

    def conv_block(self, index, in_channels, out_channels, kernel_size, dilation, padding, bn_affine):
         return [('conv%d' % index, nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)),
                 ('relu%d' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%d' % index, nn.BatchNorm1d(out_channels, affine=bn_affine)),
                 ('linear%d' % index, TransposeLinear(out_channels, out_channels)),
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
            if isinstance(m, GaussLinear):
                logger.info("Initializing %s with zeros" % str(m))
            if isinstance(m, nn.Linear):
                logger.info("Initializing %s with xvavier normal" % str(m))
                #nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU

    def mean_std_pooling(self, x, eps=1e-9):
        # mean
        N = x.shape[0]
        T = x.shape[2]
        m = torch.mean(x, dim=2)

        # std
        # NOTE: std has stability issues as autograd of std(0) is NaN
        # we can use var, but that is not in ONNX export, so
        # do it the long way
        #s = torch.sqrt(torch.var(x, dim=2) + eps)
        s = torch.sqrt(torch.mean((x - m.unsqueeze(2))**2, dim=2) + eps)

        x = torch.cat([m, s], dim=1)

        return x


    def extract_embedding(self, x):
        x = self.prepooling_layers(x)
        x = self.mean_std_pooling(x)
        x = self.embedding(x)
        return x

    def forward(self, x, labels):
        # Compute embeddings 
        x = self.extract_embedding(x)        
        x = self.arc_logit(x, labels)
        return x

    def freeze_prepooling(self):
        # set model to eval mode and turn off gradients
        self.prepooling_frozen = True
        for param in self.prepooling_layers.parameters():
            param.requires_grad = False

    def freeze_embedding(self):

        # Freeze embedding and everything before it
        self.embedding_frozen = True
        self.freeze_prepooling()
        for param in self.embedding.parameters():
            param.requires_grad = False

    def train_with_freeze(self):

        # Set training mode except for frozen layers
        self.train()
        if self.prepooling_frozen:
            self.prepooling_layers.eval()
        if self.embedding_frozen:
            self.embedding.eval()






### acrmargin
class Xvector9s_arc(nn.Module):

    def __init__(self, input_dim, layer_dim, embedding_dim, num_classes, margin=0.5, scale=64.0, easy_margin=False, bn_affine=False):

        super(Xvector9s_arc, self).__init__()
        self.prepooling_frozen = False
        self.embedding_frozen = False

        layers = []
        # conv blocks
        layers.extend(self.conv_block(1, input_dim, layer_dim, 5, 1, 2, bn_affine))
        layers.extend(self.conv_block(2, layer_dim, layer_dim, 3, 2, 2, bn_affine))
        layers.extend(self.conv_block(3, layer_dim, layer_dim, 3, 3, 3, bn_affine))
        layers.extend(self.conv_block(4, layer_dim, layer_dim, 3, 4, 4, bn_affine))

        # expansion layer
        layers.extend([('expand_conv', TransposeLinear(layer_dim, layer_dim*3)),
                       ('expand_relu', nn.LeakyReLU(inplace=True)),
                       ('expand_bn', nn.BatchNorm1d(layer_dim*3, affine=bn_affine))])

        self.prepooling_layers = nn.Sequential(OrderedDict(layers))

        # pooling defined below

        # embedding
        self.embedding = nn.Linear(layer_dim*6, embedding_dim)
        
        out_layers = []
        out_layers.extend([('embedd_relu', nn.LeakyReLU(inplace=True)),
                           ('embedd_bn', nn.BatchNorm1d(embedding_dim, affine=bn_affine))])

        self.output = nn.Sequential(OrderedDict(out_layers))

        self.arc_logit = ArcMarginModel(num_classes, embedding_dim, easy_margin, margin, scale)
            
        self.init_weight()

    def conv_block(self, index, in_channels, out_channels, kernel_size, dilation, padding, bn_affine):
         return [('conv%d' % index, nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)),
                 ('relu%d' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%d' % index, nn.BatchNorm1d(out_channels, affine=bn_affine)),
                 ('linear%d' % index, TransposeLinear(out_channels, out_channels)),
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
            if isinstance(m, GaussLinear):
                logger.info("Initializing %s with zeros" % str(m))
            if isinstance(m, nn.Linear):
                logger.info("Initializing %s with xvavier normal" % str(m))
                #nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU

    def mean_std_pooling(self, x, eps=1e-9):
        # mean
        N = x.shape[0]
        T = x.shape[2]
        m = torch.mean(x, dim=2)

        # std
        # NOTE: std has stability issues as autograd of std(0) is NaN
        # we can use var, but that is not in ONNX export, so
        # do it the long way
        #s = torch.sqrt(torch.var(x, dim=2) + eps)
        s = torch.sqrt(torch.mean((x - m.unsqueeze(2))**2, dim=2) + eps)

        x = torch.cat([m, s], dim=1)

        return x


    def extract_embedding(self, x):
        x = self.prepooling_layers(x)
        x = self.mean_std_pooling(x)
        x = self.embedding(x)
        return x

    def forward(self, x, labels):
        # Compute embeddings 
        x = self.extract_embedding(x)
        x = self.output(x)     
        x = self.arc_logit(x, labels)
        return x

    def freeze_prepooling(self):

        # set model to eval mode and turn off gradients
        self.prepooling_frozen = True
        for param in self.prepooling_layers.parameters():
            param.requires_grad = False

    def freeze_embedding(self):

        # Freeze embedding and everything before it
        self.embedding_frozen = True
        self.freeze_prepooling()
        for param in self.embedding.parameters():
            param.requires_grad = False

    def train_with_freeze(self):

        # Set training mode except for frozen layers
        self.train()
        if self.prepooling_frozen:
            self.prepooling_layers.eval()
        if self.embedding_frozen:
            self.embedding.eval()



### dgr version
class Xvector9s_att(nn.Module):

    def __init__(self, input_dim, layer_dim, embedding_dim, num_classes, h_dim=128, n_heads=32, n_stages=0, bn_affine=False):

        super(Xvector9s_att, self).__init__()
        self.prepooling_frozen = False
        self.embedding_frozen = False

        layers = []
        # conv blocks
        layers.extend(self.conv_block(1, input_dim, layer_dim, 5, 1, 2, bn_affine))
        layers.extend(self.conv_block(2, layer_dim, layer_dim, 3, 2, 2, bn_affine))
        layers.extend(self.conv_block(3, layer_dim, layer_dim, 3, 3, 3, bn_affine))
        layers.extend(self.conv_block(4, layer_dim, layer_dim, 3, 4, 4, bn_affine))

        self.prepooling_layers = nn.Sequential(OrderedDict(layers))

        # multi-head attention
        self.mh_att = MH_att(layer_dim, h_dim, n_heads, n_stages, bn_affine)

        # embedding
        self.embedding = nn.Linear(h_dim*n_heads, embedding_dim)
        
        out_layers = []
        out_layers.extend([('embedd_relu', nn.LeakyReLU(inplace=True)),
                           ('embedd_bn', nn.BatchNorm1d(embedding_dim, affine=bn_affine)),
                           ('out_linear', nn.Linear(embedding_dim, num_classes))])

        self.output = nn.Sequential(OrderedDict(out_layers))        
            
        self.init_weight()

    def conv_block(self, index, in_channels, out_channels, kernel_size, dilation, padding, bn_affine=False):
         return [('conv%d' % index, nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)),
                 ('relu%d' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%d' % index, nn.BatchNorm1d(out_channels, affine=bn_affine)),
                 ('linear%d' % index, TransposeLinear(out_channels, out_channels)),
                 ('relu%da' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%da' % index, nn.BatchNorm1d(out_channels, affine=bn_affine))]


    def init_weight(self):
        """
        Initialize weight with sensible defaults for the various layer types
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                logger.info("Initializing %s with kaiming normal" % str(m))                
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU
            if isinstance(m, GaussLinear):
                logger.info("Initializing %s with zeros" % str(m))
            if isinstance(m, nn.Linear):
                logger.info("Initializing %s with kaiming normal" % str(m))                
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU


    def extract_embedding(self, x):
        x = self.prepooling_layers(x)
        x = self.mh_att(x)
        x = self.embedding(x)
        return x

    def forward(self, x):
        # Compute embeddings 
        x = self.extract_embedding(x)
        x = self.output(x)        
        return x

    def freeze_prepooling(self):
        # set model to eval mode and turn off gradients
        self.prepooling_frozen = True
        for param in self.prepooling_layers.parameters():
            param.requires_grad = False

    def freeze_embedding(self):
        # Freeze embedding and everything before it
        self.embedding_frozen = True
        self.freeze_prepooling()
        for param in self.embedding.parameters():
            param.requires_grad = False

    def train_with_freeze(self):
        # Set training mode except for frozen layers
        self.train()
        if self.prepooling_frozen:
            self.prepooling_layers.eval()
        if self.embedding_frozen:
            self.embedding.eval()


### dgr with affine_true BN
class Xvector9s_dgr_affine_bn(nn.Module):

    def __init__(self, input_dim, layer_dim, embedding_dim, num_classes):

        super(Xvector9s_dgr_affine_bn, self).__init__()
        self.prepooling_frozen = False
        self.embedding_frozen = False

        layers = []
        # conv blocks
        layers.extend(self.conv_block(1, input_dim, layer_dim, 5, 1, 2))
        layers.extend(self.conv_block(2, layer_dim, layer_dim, 3, 2, 2))
        layers.extend(self.conv_block(3, layer_dim, layer_dim, 3, 3, 3))
        layers.extend(self.conv_block(4, layer_dim, layer_dim, 3, 4, 4))

        # expansion layer
        layers.extend([('expand_conv', TransposeLinear(layer_dim, layer_dim*3)),
                       ('expand_relu', nn.LeakyReLU(inplace=True)),
                       ('expand_bn', nn.BatchNorm1d(layer_dim*3, affine=True))])

        self.prepooling_layers = nn.Sequential(OrderedDict(layers))

        # pooling defined below

        # embedding
        self.embedding = nn.Linear(layer_dim*6, embedding_dim)
        
        out_layers = []
        out_layers.extend([('embedd_relu', nn.LeakyReLU(inplace=True)),
                           ('embedd_bn', nn.BatchNorm1d(embedding_dim, affine=True)),
                           ('out_linear', nn.Linear(embedding_dim, num_classes))])

        self.output = nn.Sequential(OrderedDict(out_layers))
        #self.output = nn.Linear(embedding_dim, num_classes)
            
        self.init_weight()

    def conv_block(self, index, in_channels, out_channels, kernel_size, dilation, padding):
         return [('conv%d' % index, nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)),
                 ('relu%d' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%d' % index, nn.BatchNorm1d(out_channels, affine=True)),
                 ('linear%d' % index, TransposeLinear(out_channels, out_channels)),
                 ('relu%da' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%da' % index, nn.BatchNorm1d(out_channels, affine=True))]


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
            if isinstance(m, GaussLinear):
                logger.info("Initializing %s with zeros" % str(m))
            if isinstance(m, nn.Linear):
                logger.info("Initializing %s with xvavier normal" % str(m))
                #nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU

    def mean_std_pooling(self, x, eps=1e-9):
        # mean
        N = x.shape[0]
        T = x.shape[2]
        m = torch.mean(x, dim=2)

        # std
        # NOTE: std has stability issues as autograd of std(0) is NaN
        # we can use var, but that is not in ONNX export, so
        # do it the long way
        #s = torch.sqrt(torch.var(x, dim=2) + eps)
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

    def freeze_prepooling(self):

        # set model to eval mode and turn off gradients
        self.prepooling_frozen = True
        for param in self.prepooling_layers.parameters():
            param.requires_grad = False

    def freeze_embedding(self):

        # Freeze embedding and everything before it
        self.embedding_frozen = True
        self.freeze_prepooling()
        for param in self.embedding.parameters():
            param.requires_grad = False

    def train_with_freeze(self):

        # Set training mode except for frozen layers
        self.train()
        if self.prepooling_frozen:
            self.prepooling_layers.eval()
        if self.embedding_frozen:
            self.embedding.eval()



### dgr 2x expansion

class Xvector9s_dgr_var_expansion(nn.Module):

    def __init__(self, input_dim, layer_dim, embedding_dim, num_classes, expansion_rate=3, drop_p=0.0):

        super(Xvector9s_dgr_var_expansion, self).__init__()
        self.prepooling_frozen = False
        self.embedding_frozen = False

        layers = []
        # conv blocks
        layers.extend(self.conv_block(1, input_dim, layer_dim, 5, 1, 2))
        layers.extend(self.conv_block(2, layer_dim, layer_dim, 3, 2, 2))
        layers.extend(self.conv_block(3, layer_dim, layer_dim, 3, 3, 3))
        layers.extend(self.conv_block(4, layer_dim, layer_dim, 3, 4, 4))

        # expansion layer
        layers.extend([('expand_conv', TransposeLinear(layer_dim, layer_dim*expansion_rate)),
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
                           ('embedd_bn', nn.BatchNorm1d(embedding_dim, affine=False)),
                           ('out_linear', nn.Linear(embedding_dim, num_classes))])

        self.output = nn.Sequential(OrderedDict(out_layers))
            
        self.init_weight()

    def conv_block(self, index, in_channels, out_channels, kernel_size, dilation, padding):
         return [('conv%d' % index, nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)),
                 ('relu%d' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%d' % index, nn.BatchNorm1d(out_channels, affine=False)),
                 ('linear%d' % index, TransposeLinear(out_channels, out_channels)),
                 ('relu%da' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%da' % index, nn.BatchNorm1d(out_channels, affine=False))]


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
            if isinstance(m, GaussLinear):
                logger.info("Initializing %s with zeros" % str(m))
            if isinstance(m, nn.Linear):
                logger.info("Initializing %s with xvavier normal" % str(m))
                #nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU

    def mean_std_pooling(self, x, eps=1e-9):
        # mean
        N = x.shape[0]
        T = x.shape[2]
        m = torch.mean(x, dim=2)

        # std
        # NOTE: std has stability issues as autograd of std(0) is NaN
        # we can use var, but that is not in ONNX export, so
        # do it the long way
        #s = torch.sqrt(torch.var(x, dim=2) + eps)
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

    def freeze_prepooling(self):

        # set model to eval mode and turn off gradients
        self.prepooling_frozen = True
        for param in self.prepooling_layers.parameters():
            param.requires_grad = False

    def freeze_embedding(self):

        # Freeze embedding and everything before it
        self.embedding_frozen = True
        self.freeze_prepooling()
        for param in self.embedding.parameters():
            param.requires_grad = False

    def train_with_freeze(self):

        # Set training mode except for frozen layers
        self.train()
        if self.prepooling_frozen:
            self.prepooling_layers.eval()
        if self.embedding_frozen:
            self.embedding.eval()



### dgr version
class Xvector9s_dgr(nn.Module):

    def __init__(self, input_dim, layer_dim, embedding_dim, num_classes):

        super(Xvector9s_dgr, self).__init__()
        self.prepooling_frozen = False
        self.embedding_frozen = False

        layers = []
        # conv blocks
        layers.extend(self.conv_block(1, input_dim, layer_dim, 5, 1, 2))
        layers.extend(self.conv_block(2, layer_dim, layer_dim, 3, 2, 2))
        layers.extend(self.conv_block(3, layer_dim, layer_dim, 3, 3, 3))
        layers.extend(self.conv_block(4, layer_dim, layer_dim, 3, 4, 4))

        # expansion layer
        layers.extend([('expand_conv', TransposeLinear(layer_dim, layer_dim*3)),
                       ('expand_relu', nn.LeakyReLU(inplace=True)),
                       ('expand_bn', nn.BatchNorm1d(layer_dim*3, affine=False))])

        self.prepooling_layers = nn.Sequential(OrderedDict(layers))

        # pooling defined below

        # embedding
        self.embedding = nn.Linear(layer_dim*6, embedding_dim)
        
        out_layers = []
        out_layers.extend([('embedd_relu', nn.LeakyReLU(inplace=True)),
                           ('embedd_bn', nn.BatchNorm1d(embedding_dim, affine=False)),
                           ('out_linear', nn.Linear(embedding_dim, num_classes))])

        self.output = nn.Sequential(OrderedDict(out_layers))
        #self.output = nn.Linear(embedding_dim, num_classes)
            
        self.init_weight()

    def conv_block(self, index, in_channels, out_channels, kernel_size, dilation, padding):
         return [('conv%d' % index, nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)),
                 ('relu%d' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%d' % index, nn.BatchNorm1d(out_channels, affine=False)),
                 ('linear%d' % index, TransposeLinear(out_channels, out_channels)),
                 ('relu%da' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%da' % index, nn.BatchNorm1d(out_channels, affine=False))]


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
            if isinstance(m, GaussLinear):
                logger.info("Initializing %s with zeros" % str(m))
            if isinstance(m, nn.Linear):
                logger.info("Initializing %s with xvavier normal" % str(m))
                #nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU

    def mean_std_pooling(self, x, eps=1e-9):
        # mean
        N = x.shape[0]
        T = x.shape[2]
        m = torch.mean(x, dim=2)

        # std
        # NOTE: std has stability issues as autograd of std(0) is NaN
        # we can use var, but that is not in ONNX export, so
        # do it the long way
        #s = torch.sqrt(torch.var(x, dim=2) + eps)
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

    def freeze_prepooling(self):

        # set model to eval mode and turn off gradients
        self.prepooling_frozen = True
        for param in self.prepooling_layers.parameters():
            param.requires_grad = False

    def freeze_embedding(self):

        # Freeze embedding and everything before it
        self.embedding_frozen = True
        self.freeze_prepooling()
        for param in self.embedding.parameters():
            param.requires_grad = False

    def train_with_freeze(self):

        # Set training mode except for frozen layers
        self.train()
        if self.prepooling_frozen:
            self.prepooling_layers.eval()
        if self.embedding_frozen:
            self.embedding.eval()



##### Fast xvec
### dgr version
class Xvector9s_fast(nn.Module):

    def __init__(self, input_dim, layer_dim, embedding_dim, num_classes):

        super(Xvector9s_fast, self).__init__()
        self.prepooling_frozen = False
        self.embedding_frozen = False

        layers = []
        # conv blocks
        layers.extend(self.conv_block(1, input_dim, layer_dim, 5, 1, 2))
        layers.extend(self.conv_block(2, layer_dim, layer_dim, 3, 2, 2))
        layers.extend(self.conv_block(3, layer_dim, layer_dim, 3, 3, 3))
        layers.extend(self.conv_block(4, layer_dim, layer_dim, 3, 4, 4))

        # expansion layer
        layers.extend([('expand_conv', TransposeLinear(layer_dim, layer_dim*3)),
                       ('expand_bn', nn.BatchNorm1d(layer_dim*3, affine=True)),
                       ('expand_relu', nn.LeakyReLU(inplace=True))])

        self.prepooling_layers = nn.Sequential(OrderedDict(layers))

        # pooling defined below

        # embedding
        self.embedding = nn.Linear(layer_dim*6, embedding_dim)
        
        out_layers = []
        out_layers.extend([('embedd_bn', nn.BatchNorm1d(embedding_dim, affine=True)),
                           ('embedd_relu', nn.LeakyReLU(inplace=True)),
                           ('out_linear', nn.Linear(embedding_dim, num_classes))])

        self.output = nn.Sequential(OrderedDict(out_layers))
            
        self.init_weight()

    def conv_block(self, index, in_channels, out_channels, kernel_size, dilation, padding):
         return [('conv%d' % index, nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)),
                 ('bn%d' % index, nn.BatchNorm1d(out_channels, affine=True)),
                 ('relu%d' % index, nn.LeakyReLU(inplace=True)),
                 ('linear%d' % index, TransposeLinear(out_channels, out_channels)),
                 ('bn%da' % index, nn.BatchNorm1d(out_channels, affine=True)),
                 ('relu%da' % index, nn.LeakyReLU(inplace=True))]

    def init_weight(self):
        """
        Initialize weight with sensible defaults for the various layer types
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                logger.info("Initializing %s with kaiming normal" % str(m))
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU
            if isinstance(m, nn.Linear):
                logger.info("Initializing %s with kaiming normal" % str(m))
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU

    def mean_std_pooling(self, x, eps=1e-9):
        # mean
        N = x.shape[0]
        T = x.shape[2]
        m = torch.mean(x, dim=2)

        # std
        # NOTE: std has stability issues as autograd of std(0) is NaN
        # we can use var, but that is not in ONNX export, so
        # do it the long way
        #s = torch.sqrt(torch.var(x, dim=2) + eps)
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

    def freeze_prepooling(self):

        # set model to eval mode and turn off gradients
        self.prepooling_frozen = True
        for param in self.prepooling_layers.parameters():
            param.requires_grad = False

    def freeze_embedding(self):

        # Freeze embedding and everything before it
        self.embedding_frozen = True
        self.freeze_prepooling()
        for param in self.embedding.parameters():
            param.requires_grad = False

    def train_with_freeze(self):

        # Set training mode except for frozen layers
        self.train()
        if self.prepooling_frozen:
            self.prepooling_layers.eval()
        if self.embedding_frozen:
            self.embedding.eval()





####  Synchronized BN
### dgr version
class Xvector9s_dgr_syncbn(nn.Module):

    def __init__(self, input_dim, layer_dim, embedding_dim, num_classes):

        super(Xvector9s_dgr_syncbn, self).__init__()
        self.prepooling_frozen = False
        self.embedding_frozen = False

        layers = []
        # conv blocks
        layers.extend(self.conv_block(1, input_dim, layer_dim, 5, 1, 2))
        layers.extend(self.conv_block(2, layer_dim, layer_dim, 3, 2, 2))
        layers.extend(self.conv_block(3, layer_dim, layer_dim, 3, 3, 3))
        layers.extend(self.conv_block(4, layer_dim, layer_dim, 3, 4, 4))

        # expansion layer
        layers.extend([('expand_conv', TransposeLinear(layer_dim, layer_dim*3)),
                       ('expand_relu', nn.LeakyReLU(inplace=True)),
                       ('expand_bn', SynchronizedBatchNorm1d(layer_dim*3, affine=False))])

        self.prepooling_layers = nn.Sequential(OrderedDict(layers))

        # pooling defined below

        # embedding
        self.embedding = nn.Linear(layer_dim*6, embedding_dim)
        
        out_layers = []
        out_layers.extend([('embedd_relu', nn.LeakyReLU(inplace=True)),
                           ('embedd_bn', SynchronizedBatchNorm1d(embedding_dim, affine=False)),
                           ('out_linear', nn.Linear(embedding_dim, num_classes))])

        self.output = nn.Sequential(OrderedDict(out_layers))
        #self.output = nn.Linear(embedding_dim, num_classes)
            
        self.init_weight()

    def conv_block(self, index, in_channels, out_channels, kernel_size, dilation, padding):
         return [('conv%d' % index, nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)),
                 ('relu%d' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%d' % index, SynchronizedBatchNorm1d(out_channels, affine=False)),
                 ('linear%d' % index, TransposeLinear(out_channels, out_channels)),
                 ('relu%da' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%da' % index, SynchronizedBatchNorm1d(out_channels, affine=False))]


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
            if isinstance(m, GaussLinear):
                logger.info("Initializing %s with zeros" % str(m))
            if isinstance(m, nn.Linear):
                logger.info("Initializing %s with kaiming normal" % str(m))
                #nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU

    def mean_std_pooling(self, x, eps=1e-9):
        # mean
        N = x.shape[0]
        T = x.shape[2]
        m = torch.mean(x, dim=2)

        # std
        # NOTE: std has stability issues as autograd of std(0) is NaN
        # we can use var, but that is not in ONNX export, so
        # do it the long way
        #s = torch.sqrt(torch.var(x, dim=2) + eps)
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

    def freeze_prepooling(self):

        # set model to eval mode and turn off gradients
        self.prepooling_frozen = True
        for param in self.prepooling_layers.parameters():
            param.requires_grad = False

    def freeze_embedding(self):

        # Freeze embedding and everything before it
        self.embedding_frozen = True
        self.freeze_prepooling()
        for param in self.embedding.parameters():
            param.requires_grad = False

    def train_with_freeze(self):

        # Set training mode except for frozen layers
        self.train()
        if self.prepooling_frozen:
            self.prepooling_layers.eval()
        if self.embedding_frozen:
            self.embedding.eval()



####
class Xvector9s(nn.Module):
    # LL can be linear or Gauss
    def __init__(self, input_dim, layer_dim, embedding_dim, num_classes, LL='linear', T0=0.0, N0=9, fixed_N=True, length_norm=False, r=0.9, enroll_type='Bayes'):

        super(Xvector9s, self).__init__()
        self.LL = LL
        self.T0 = T0 # duration model
        if T0:
            logger.info("Duration modeling with T0=%.2f",T0)
        self.enroll_type = enroll_type
        self.r = r
        self.N_dict = {}
        self.prepooling_frozen = False
        self.embedding_frozen = False
        layers = []

        # conv blocks
        layers.extend(self.conv_block(1, input_dim, layer_dim, 5, 1, 2))
        layers.extend(self.conv_block(2, layer_dim, layer_dim, 3, 2, 2))
        layers.extend(self.conv_block(3, layer_dim, layer_dim, 3, 3, 3))
        layers.extend(self.conv_block(4, layer_dim, layer_dim, 3, 4, 4))

        # expansion layer
        layers.extend([('expand_conv', TransposeLinear(layer_dim, layer_dim*3)),
                       ('expand_relu', nn.LeakyReLU(inplace=True)),
                       ('expand_bn', nn.BatchNorm1d(layer_dim*3))])

        self.prepooling_layers = nn.Sequential(OrderedDict(layers))

        # pooling defined below

        # embedding
        #self.embedding = nn.Linear(layer_dim*6, embedding_dim, bias=False)
        self.embedding = nn.Linear(layer_dim*6, embedding_dim, bias=True)

        # length-norm and PLDA layer
        self.plda = plda(embedding_dim, num_classes, length_norm)

        # output for Gauss
        self.output = None
        if self.LL == 'Gauss':
            #self.output = GaussLinear(embedding_dim, num_classes, N0, fixed_N, discr_mean=False)
            self.output = GaussQuadratic(embedding_dim, num_classes, self.plda, N0, fixed_N, r, enroll_type, self.N_dict)

        elif self.LL == 'Gauss_discr':
            self.output = GaussLinear(embedding_dim, num_classes, N0, fixed_N, discr_mean=True)

        elif self.LL == 'linear':
            self.output = nn.Linear(embedding_dim, num_classes)

        self.init_weight()

    def conv_block(self, index, in_channels, out_channels, kernel_size, dilation, padding):
         return [('conv%d' % index, nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)),
                 ('bn%d' % index, nn.BatchNorm1d(out_channels)),
                 ('relu%d' % index, nn.LeakyReLU(inplace=True)),
                 ('linear%d' % index, TransposeLinear(out_channels, out_channels)),
                 ('bn%da' % index, nn.BatchNorm1d(out_channels)),
                 ('relu%da' % index, nn.LeakyReLU(inplace=True))]


    def init_weight(self):
        """
        Initialize weight with sensible defaults for the various layer types
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                logger.info("Initializing %s with xvaivier normal" % str(m))
                nn.init.xavier_normal_(m.weight)
            if isinstance(m, GaussLinear):
                logger.info("Initializing %s with zeros" % str(m))
            elif isinstance(m, nn.Linear):
                logger.info("Initializing %s with xvavier normal" % str(m))
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                logger.info("Initializing %s with constant (1,. 0)" % str(m))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def mean_std_pooling(self, x, eps=1e-9):
        # mean
        N = x.shape[0]
        T = x.shape[2]
        m = torch.mean(x, dim=2)

        # std
        # NOTE: std has stability issues as autograd of std(0) is NaN
        # we can use var, but that is not in ONNX export, so
        # do it the long way
        #s = torch.sqrt(torch.var(x, dim=2) + eps)
        s = torch.sqrt(torch.mean((x - m.unsqueeze(2))**2, dim=2) + eps)

        x = torch.cat([m, s], dim=1)

        # confidence weight based on duration
        w = x.new_zeros((N,))
        w[:] = T / float(T + self.T0)
        return x, w


    def extract_embedding(self, x):
        x = self.prepooling_layers(x)
        x, w = self.mean_std_pooling(x)
        x = self.embedding(x)
        return x, w

    def forward(self, x, embedding_only=False):

        # Compute embeddings 
        x, w = self.extract_embedding(x)

        # Length norm and PLDA
        y, z = self.plda(x)

        # Compute output class log likelihoods
        if embedding_only or self.output is None:
            LL = None
        elif self.LL == 'Gauss_discr' or self.LL == 'linear':
            # Discriminative outer layer: take pre-LDA output
            LL = self.output(y) / w[:,None]
        else:
            # Gaussian after PLDA
            LL = self.output(z) / w[:,None]

        return x, y, z, LL, w

    def freeze_prepooling(self):

        # set model to eval mode and turn off gradients
        self.prepooling_frozen = True
        for param in self.prepooling_layers.parameters():
            param.requires_grad = False

    def freeze_embedding(self):

        # Freeze embedding and everything before it
        self.embedding_frozen = True
        self.freeze_prepooling()
        for param in self.embedding.parameters():
            param.requires_grad = False

    def train_with_freeze(self):

        # Set training mode except for frozen layers
        self.train()
        if self.prepooling_frozen:
            self.prepooling_layers.eval()
        if self.embedding_frozen:
            self.embedding.eval()

    def update_params(self, x, y, z, labels):

        self.plda.update_center(x)
        self.plda.update_plda(y, labels)
        if self.LL == 'Gauss' or self.LL == 'Gauss_discr':
            self.output.update_params(y, labels)
        return

# length-norm and PLDA layer
class plda(nn.Module):
    def __init__(self, embedding_dim, num_classes, length_norm=True, N0=30, center_l=1e-4):
        super(plda, self).__init__()

        self.N0 = N0     # running sum size for model mean updates in PLDA
        self.length_norm_flag = length_norm
        self.center = nn.Parameter(torch.zeros(embedding_dim), requires_grad=False)
        self.center_l = center_l
        self.d_wc = nn.Parameter(torch.ones(embedding_dim), requires_grad=False)
        self.d_ac = nn.Parameter(torch.ones(embedding_dim), requires_grad=False)
        self.Ulda = nn.Parameter(torch.eye(embedding_dim), requires_grad=False)
        if self.length_norm_flag:
            logger.info("Initializing length_norm scaling with sqrt(dimension)")
            self.norm_scale = nn.Parameter(torch.sqrt(torch.tensor(float(embedding_dim))), requires_grad=True)
        else:
            self.norm_scale = None
        self.register_buffer('sums', torch.zeros(num_classes,embedding_dim))
        self.register_buffer('counts', torch.zeros(num_classes,))
        self.plda_cnt = 0

    def forward(self, x):

        # Length norm (optional)
        if self.length_norm_flag:
            y = self.length_norm(x)
        else:
            y = x

        # LDA diagonalization, (U^t*y^t)^t = y*U
        z = torch.mm(y,self.Ulda)
        return y, z

    def length_norm(self, x):
        y = self.norm_scale*F.normalize(x, p=2, dim=1)
        return y

    # Cost of norm: average magnitude should be one
    def norm_loss(self, x):

        y = x - x.mean(dim=0)
        loss = (1.0-torch.sqrt(torch.mean(y**2)))**2
        return loss

    # Cost of center
    def center_loss(self, x):

        m = x.mean(dim=0)
        if 0:
            l = self.center_l
            m0 = self.center
            loss = ((1-l)**2)*torch.sum(m0**2) + 2*(1-l)*l*torch.sum(m0*m) + (l**2)*torch.sum(m**2)
        else:
            loss = torch.mean(m**2)
        return loss

    def update_center(self, x):

        # Update mean estimate from data
        with torch.no_grad():
            self.center.data = (1-self.center_l)*self.center + self.center_l*x.mean(dim=0)

    def update_plda(self, x, labels):

        # Update across-class covariance based on sample means
        plda_ds = 0
        with torch.no_grad():

            update_counts(x, labels, self.sums, self.counts, self.N0)
            self.plda_cnt -= 1
            if self.counts.min() > 1 and self.plda_cnt <= 0:
                self.plda_cnt = plda_ds
                means = self.sums/self.counts[:,None]
                y = means - means.mean(dim=0)
                cov_ac = torch.mm(y.t(),y) / (means.shape[0])

                # Simultaneous diagonalization of wc and ac covariances
                # Assume wc=I
                if 1:
                    # Brute force every time
                    eval, self.Ulda.data = torch.symeig(cov_ac, eigenvectors=True)
                else:
                    # Try for continuity vs. previous eigendecomposition
                    if 0:
                        # Start with existing diagonalization and update
                        S = torch.mm(torch.mm(torch.t(self.Ulda),cov_ac),self.Ulda)
                        ev,U2 = torch.symeig(S, eigenvectors=True)
                        U = torch.mm(self.Ulda,U2)
                        # keep orthogonal
                        if 0:
                            I = torch.eye(cov_ac.shape[0],device=cov_ac.device)
                            U2 = torch.mm(U, I + 0.5*(I-torch.mm(torch.t(U),U)))
                    else:
                        eval, U2 = torch.symeig(cov_ac, eigenvectors=True)
                    R = torch.mm(torch.t(U2),self.Ulda)
                    d = R.shape[0]
                    tmp = torch.topk(torch.abs(R.view(d**2)),d+2)
                    rot = {}
                    for d2 in range(d-1,d+2):
                        thresh = 0.5*(tmp[0][d2-1]+tmp[0][d2])
                        rot[d2] = 0*R
                        rot[d2][R>=thresh] = 1
                        rot[d2][-R>thresh] = -1
                    Uold = 1.0*self.Ulda
                    r2 = rot[d]
                    Rfound = False
                    for n in range(2):
                        N0 = (0.5+torch.sum(torch.abs(r2)).cpu().numpy()).astype(np.int)
                        if not N0 == d:
                            print("Rotation wrong sum %d" % N0)
                        if (N0 == d) and (torch.max(torch.mm(torch.t(r2),r2)) < 1.5):
                            self.Ulda.data = torch.mm(U2,r2)
                            Rfound = True
                            print("Rotation successful on try %d, N0 = %d" %(n,N0))
                            break
                        # Try to fix with next best candidate
                        r2 = rot[d-1] + (rot[d+1]-rot[d])
                    if not Rfound:
                        print("WARNING: Rotation failed.")
                        print(torch.max(torch.mm(torch.t(rot[d]),rot[d])))
                        print(torch.sum(torch.abs(rot[d])))
                        print(thresh)
                        print(tmp[0][d-2:d+2])
                        self.Ulda.data = U2
                                        
                self.d_ac.data = torch.diag(torch.mm(torch.mm(torch.t(self.Ulda),cov_ac),self.Ulda))


# Function to update running sums and counts for classes
def update_counts(x, labels, sums, counts, N0, rand_flag=False):

    N = x.shape[0]
    for n in range(N):
        m = labels[n]
        N1 = N0
        if counts[m] > (N0-0.5) and rand_flag:
            # Random reset to 1 with probability 1/N0
            if ((counts[m] > (2*N0)-0.5) or random.randint(1,N0) == 1):
                N1 = 1
            else:
                N1 = 2*N0
        if counts[m] < (N1-0.5):
            # Running sum at first
            sums[m,:] += x[n,:]
            counts[m] += 1
        else:
            # Recursive sum and count for forgetting
            sums[m,:] = (N1-1)*sums[m,:]/counts[m] + x[n,:]
            counts[m] = N1
        if m == 100:
            logger.info(" spkr100 count %.2f N0 %2d N1 %2d", counts[m], N0, N1)
            logger.info(" spkr101 count %.2f", counts[m+1])


class GaussLinear(nn.Module):
    def __init__(self, embedding_dim, num_classes, N0=9, fixed_N=True, discr_mean=False):
        super(GaussLinear, self).__init__()

        self.N0 = N0     # running sum size for mean updates (enrollment)
        self.fixed_N = fixed_N # fixed or random N0 per batch
        self.discr_mean = discr_mean
        if self.discr_mean:
            # Discriminative training of means but bias based on formula
            self.means = nn.Parameter(torch.zeros(num_classes,embedding_dim), requires_grad=True)
        else:
            # Generative means
            self.means = nn.Parameter(torch.zeros(num_classes,embedding_dim), requires_grad=False)
        self.register_buffer('sums', torch.zeros(num_classes,embedding_dim))
        self.register_buffer('counts', torch.zeros(num_classes,))


    def forward(self, input):

        bias = -0.5*((self.means**2).sum(dim=1))
        return F.linear(input, self.means, bias)

    def update_params(self, x, labels):

        # Update running counters and means for Gaussian last layer
        with torch.no_grad():
            update_counts(x, labels, self.sums, self.counts, self.N0, rand_flag=(not self.fixed_N))

            if not self.discr_mean:
                # Generative means
                # Compute means for classes in batch
                classes = list(set(labels.tolist()))
                ind = torch.tensor(classes,device=x.device)
                self.means.data[ind,:] = self.sums[ind,:] / self.counts[ind,None]

    def mean_loss(self):
        means = self.sums / self.counts[:,None]
        loss = torch.mean((self.means-means)**2)
        return loss
        
class GaussQuadratic(nn.Module):
    def __init__(self, embedding_dim, num_classes, plda, N0=9, fixed_N=True, r=0.9, enroll_type='Bayes', N_dict={}):
        super(GaussQuadratic, self).__init__()

        self.num_classes = num_classes
        self.plda = plda
        self.N0 = N0     # running sum size for mean updates (enrollment)
        self.fixed_N = fixed_N # fixed or random N0 per batch
        self.r = r # cut correlation
        self.enroll_type = enroll_type
        self.means = nn.Parameter(torch.zeros(num_classes,embedding_dim), requires_grad=False)
        self.cov = nn.Parameter(torch.zeros(num_classes,embedding_dim), requires_grad=False)
        self.register_buffer('sums', torch.zeros(num_classes,embedding_dim))
        self.register_buffer('counts', torch.zeros(num_classes,))
        self.N_dict = N_dict

    def forward(self, x):

        # Update models and compute Gaussian log-likelihoods
        with torch.no_grad():
            self.means.data, self.cov.data = gmm_adapt(self.counts, torch.mm(self.sums,self.plda.Ulda), self.plda.d_wc, self.plda.d_ac, self.r, self.enroll_type, self.N_dict)

        N = x.shape[0]
        M = self.num_classes
        LL = x.new_zeros((N,M))
        cov_test = 1.0
        LL = gmm_score(x, self.means, self.cov+cov_test)

        return LL

    def update_params(self, x, labels):

       # Update running counters and means for Gaussian last layer
        with torch.no_grad():
            update_counts(x, labels, self.sums, self.counts, self.N0, rand_flag=(not self.fixed_N))


def prototypical_loss(input, target, n_support=1):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')


    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    print('n_classes',n_classes,'n_support',n_support,'n_query',n_query)

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    #query_samples = input.to('cpu')[query_idxs]
    query_samples = input_cpu[query_idxs]
    #dists = -euclidean_dist(query_samples, prototypes)
    dists = torch.mm(query_samples,prototypes.t())
    #nrm_proto = torch.norm(prototypes, dim=1) + 1e-8
    #nrm_query = torch.norm(query_samples, dim=1) + 1e-8

    #dists = scale*(dists / nrm_proto) / nrm_query.view(-1,1)
    
    log_p_y = F.log_softmax(dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes, device=input_cpu.device)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    C0 = torch.log(torch.tensor(n_classes,dtype=loss_val.dtype,device=loss_val.device))
    norm_loss = loss_val / C0
    #print(C0,n_classes)

    return loss_val, norm_loss, [[100.0*acc_val]]




# Wrapper function for normalized cross entropy loss and accuracy
#  can do outer layer or gaussian minibatch means
def ComputeLoss(x, y, output, w, labels, loss_type='CE', model=None):

    if loss_type == 'CE':
        loss, nloss = NCE_loss(output, labels)
        acc = accuracy(output, labels)

    elif loss_type == 'Proto':
        loss, nloss, acc = prototypical_loss(output, labels)

    elif loss_type == 'GaussLoss':
        loss, nloss, acc = GaussLoss(y, w, labels, cov_ac=model.plda.d_ac, enroll_type=model.enroll_type, r=model.r, N_dict=model.N_dict)
    else:
        raise ValueError("Invalid loss type %s." % loss_type)


    return loss, nloss, acc

# Normalized multiclass cross-entropy
def NCE_loss(LL, labels, prior=None):

    M = LL.shape[1]
    if prior is None:
        # No prior: flat over number of classes
        C0 = torch.log(torch.tensor(M,dtype=LL.dtype,device=LL.device))
    else:
        # Prior given
        C0 = -torch.sum(prior*torch.log(prior))

    if M > 1:
        loss = F.cross_entropy(LL, labels)
        nloss = (1.0/C0)*loss
    else:
        loss = torch.tensor(0.0)
        nloss = loss
    return loss, nloss

# Gaussian diarization loss in minibatch
# Compute Gaussian cost across minibatch of samples vs. average
def GaussMinibatchLL(x, w, labels, loo_flag=True, cov_wc=None, cov_ac=None, enroll_type='Bayes', r=0.9, N_dict=None):

    N = x.shape[0]
    d = x.shape[1]
    classes = list(set(labels.tolist()))
    M = len(classes)
    l2 = labels.clone()
    sums = x.new_zeros((M,d))
    counts = x.new_zeros((M,))
    means = x.new_zeros((M,d))
    cov = x.new_zeros((M,d))
    LL = x.new_zeros((N,M))
    if cov_wc is None:
        cov_wc = x.new_ones((d,))
    if cov_ac is None or len(cov_ac.shape) > 1:
        cov_ac = x.new_ones((d,))
    cov_test = 1.0
    if N_dict is None:
        N_dict = {}

    # Compute stats for classes (separate loop since in-place update)
    for m in range(M):
        m2 = classes[m]
        ind = torch.tensor(labels==m2,device=x.device)
        l2[ind] = m
        sums[m,:] += (x[ind,:]*w[ind,None]).sum(dim=0)
        counts[m] += w[ind].sum()

    # Compute models and log-likelihoods
    means, cov = gmm_adapt(counts, sums, cov_wc, cov_ac, r, enroll_type, N_dict)
    LL = gmm_score(x, means, cov+cov_test)

    # Leave one out corrections
    if loo_flag:
        for n in range(N):
            m = classes.index(labels[n])
            if counts[m]-w[n] > 1e-8:
                mu_model, cov_model = gmm_adapt(counts[m:m+1]-w[n], sums[m:m+1,:]-w[n]*x[n,:], cov_wc, cov_ac, r, enroll_type, N_dict)
                LL[n,m] = gmm_score(x[n:n+1,:], mu_model, cov_model+cov_test)

    # Compute and apply prior
    #LL = LL/w[:,None]
    prior = counts/counts.sum()
    logprior = torch.log(prior)
    LL += logprior

    return LL, prior, l2

# Gaussian diarization loss in minibatch
def GaussLoss(x, w, labels, loo_flag=True, cov_wc=None, cov_ac=None, enroll_type='Bayes', r=0.9, N_dict=None):

    # Return normalized cross entropy cost
    LL, prior, l2 = GaussMinibatchLL(x, w, labels, loo_flag, cov_wc, cov_ac, enroll_type, r, N_dict)
    loss, nloss = NCE_loss(LL, l2, prior)
    acc = accuracy(LL,l2)
    return loss, nloss, acc

# Function for Bayesian adaptation of Gaussian model
# Enroll type can be ML, MAP, or Bayes
def gmm_adapt(cnt, xsum, cov_wc, cov_ac, r=0, enroll_type='ML', N_dict=None):

    # Compute ML model
    cnt = torch.max(0*cnt+(1e-10),cnt)
    mu_model = xsum / cnt[:,None]
    cov_model = 0*mu_model

    if not enroll_type == 'ML':

        # MAP adaptation
        # Determine covariance of model mean posterior distribution
        # Determine mean of model mean posterior distribution

        if r == 0:
            Nsc = 1.0/cnt
        elif r == 1:
            Nsc = 0.0*cnt+1.0
        else:
            Nsc = compute_Nsc(cnt, r, N_dict)
        cov_mean = cov_wc*Nsc[:,None]

        # MAP mean plus model uncertainty
        temp = cov_ac / (cov_ac + cov_mean)
        mu_model *= temp
        if enroll_type == 'Bayes':
            # Bayesian covariance of mean uncertainty
            cov_model = temp*cov_mean

    # Return
    return mu_model, cov_model

def compute_Nsc(cnts, r, N_dict=None):

    # Correlation model for enrollment cuts (0=none,1=single-cut)
    if N_dict is None:
        N_dict = {}
    Nsc = cnts.clone()
    icnt = (0.5+cnts.cpu().numpy()).astype(np.int)
    for cnt in np.unique(icnt):
        if cnt not in N_dict.keys():
            #print("cnt not in dict", cnt)
            Nm = max(1,int(cnt+0.5))
            temp = 1.0
            for T in range(1,Nm):
                temp += 2.0*(r**T)*(Nm-T)/Nm
            temp = temp/Nm

            # Apply integer count change to float value
            N_dict[cnt] = float(temp*Nm) / max(1e-10,cnt)
        
        # Update N_eff
        Nsc[cnts==cnt] = N_dict[cnt]

    return Nsc

def gmm_score(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model"""

    inv_covars = 1.0/covars
    n_samples, n_dim = X.shape
    LLs = -0.5 * (- torch.sum(torch.log(inv_covars), 1)
                  + torch.sum((means ** 2) * inv_covars, 1)
                  - 2 * torch.mm(X, torch.t(means * inv_covars)))
    LLs -= 0.5 * (torch.mm(X ** 2, torch.t(inv_covars)))
    #LLs -= 0.5 * (n_dim * np.log(2 * np.pi))

    return LLs




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class angular_ResNet(nn.Module):

    def __init__(self, input_dim, num_classes, block=BasicBlock, layers=[3, 4, 6, 3],  embedding_dim=256, inplanes=32, zero_init_residual=True, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, margin=0.2, scale=30.0, easy_margin=False):
        super(angular_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, 2*inplanes, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 4*inplanes, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 8*inplanes, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
                
        # Use mean_std_dev_pooling
        
        # Compute embedding
        self.embedding = nn.Linear((input_dim//4) * 8*inplanes, embedding_dim)

        self.arc_logit = ArcMarginModel(num_classes, embedding_dim, easy_margin, margin, scale)
        self.base_margin = margin


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def mean_std_pooling(self, x, eps=1e-9):
        m = torch.mean(x, dim=2)
        s = torch.sqrt(torch.mean((x - m.unsqueeze(2))**2, dim=2) + eps)
        x = torch.cat([m, s], dim=1)
        return x

    def extract_embedding(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)        

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.flatten(x, 1, 2)
        x = self.mean_std_pooling(x)        
        x = self.embedding(x)

        return x
    
    
    def forward(self, x, labels=torch.arange(2), epoch=1):
        # Compute embeddings                                                                                                                                                                                                                                                                                                   
        x = self.extract_embedding(x)
        if epoch < 2:
            self.arc_logit.set_margin(0.0)                                                                                                                                                                                                                                                                     
        else:
            self.arc_logit.set_margin(self.base_margin)

        x = self.arc_logit(x, labels)

        return x

        
### Resnet traditional softmax
class traditional_ResNet(nn.Module):

    def __init__(self, input_dim, num_classes, block=BasicBlock, layers=[3, 4, 6, 3],  embedding_dim=256, inplanes=32, zero_init_residual=True, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(traditional_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, 2*inplanes, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 4*inplanes, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 8*inplanes, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
                
        # Use mean_std_dev_pooling
        
        # Compute embedding
        self.embedding = nn.Linear((input_dim//4) * 8*inplanes, embedding_dim)

        out_layers = []
        out_layers.extend([('embedd_relu', nn.ReLU(inplace=True)),
                           ('embedd_bn', nn.BatchNorm1d(embedding_dim, affine=False)),
                           ('out_linear', nn.Linear(embedding_dim, num_classes))])

        self.output = nn.Sequential(OrderedDict(out_layers))


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def mean_std_pooling(self, x, eps=1e-9):
        m = torch.mean(x, dim=2)
        s = torch.sqrt(torch.mean((x - m.unsqueeze(2))**2, dim=2) + eps)
        x = torch.cat([m, s], dim=1)
        return x

    def extract_embedding(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)        

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.flatten(x, 1, 2)
        x = self.mean_std_pooling(x)        
        x = self.embedding(x)

        return x
    
    
    def forward(self, x):
        # Compute embedding                                                                                                              
        x = self.extract_embedding(x)
        x = self.output(x)
        return x
    

#### Efficient net below

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 1  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._dropout = self._global_params.dropout_rate
        #self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self.embedding = nn.Linear(out_channels*4, self._global_params.embedding_dim)
        
        self.arc_logit = ArcMarginModel(self._global_params.num_classes, self._global_params.embedding_dim, self._global_params.easy_margin, self._global_params.margin, self._global_params.scale)
        self.base_margin = self._global_params.margin

    def mean_std_pooling(self, x, eps=1e-9):
        m = torch.mean(x, dim=2)
        s = torch.sqrt(torch.mean((x - m.unsqueeze(2))**2, dim=2) + eps)
        x = torch.cat([m, s], dim=1)
        return x

    def extract_embedding(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))
        
        
        x = torch.flatten(x, 1, 2)
        x = self.mean_std_pooling(x)        
        x = self.embedding(x)
        
        

        return x


    def forward(self, x, labels=torch.arange(2), epoch=1):
        # Compute embeddings                                                                                                                                                                                    
        
        x = self.extract_embedding(x)
        if epoch < 2:
            self.arc_logit.set_margin(0.0)                                                                                                                                                                     

        else:
            self.arc_logit.set_margin(self.base_margin)

        x = self.arc_logit(x, labels)

        return x


    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        print(global_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet-b'+str(i) for i in range(num_models)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

##### Jasper


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, dropout_rate=0.0):
        """1D Convolution with the batch normalization and RELU."""
        super(ConvBlock, self).__init__()
        self.dropout_rate = dropout_rate

        if dilation > 1:
            assert stride == 1
            padding = (kernel_size - 1) * dilation // 2
        else:
            padding = (kernel_size - stride + 1) // 2

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        nn.init.kaiming_normal_(self.conv.weight)

        self.bn = nn.BatchNorm1d(out_channels)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        #y = F.relu(y)
        # OpenSeq2Seq uses max clamping instead of gradient clipping
        y = torch.clamp(y, min=0.0, max=20.0)  # like RELU but clamp at 20.0

        if self.dropout_rate > 0:
            y = F.dropout(y, p=self.dropout_rate, training=self.training)
        return y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate=0.0):
        """1D Convolution with the batch normalization and RELU."""
        super(ResidualBlock, self).__init__()
        self.dropout_rate = dropout_rate

        self.block1 = ConvBlock(in_channels, out_channels, kernel_size, dropout_rate=dropout_rate)
        self.block2 = ConvBlock(out_channels, out_channels, kernel_size, dropout_rate=dropout_rate)

        # block 3
        stride = 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        nn.init.kaiming_normal_(self.conv.weight)

        self.bn = nn.BatchNorm1d(out_channels)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

        #self.linear = TransposeLinear(in_channels, out_channels, bias=True)
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.linear.weight)

        self.bn2 = nn.BatchNorm1d(out_channels)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)


    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)

        y = self.conv(y)
        y = self.bn(y)

        p = self.linear(x)
        p = self.bn2(p)

        y += p

        #y = F.relu(y)
        # OpenSeq2Seq uses max clamping instead of gradient clipping
        y = torch.clamp(y, min=0.0, max=20.0)  # like RELU but clamp at 20.0

        if self.dropout_rate > 0:
            y = F.dropout(y, p=self.dropout_rate, training=self.training)
        return y



class Jasper(nn.Module):

    def __init__(self, vocab, feat_dim, stride=2):
        super(Jasper, self).__init__()
        self.stride = stride

        self.layers = nn.Sequential(
            ConvBlock(feat_dim, 256, 11, stride=stride, dropout_rate=0.2),

            ResidualBlock(256, 256, 11, dropout_rate=0.2),

            ResidualBlock(256, 384, 13, dropout_rate=0.2),

            ResidualBlock(384, 512, 17, dropout_rate=0.2),

            ResidualBlock(512, 640, 21, dropout_rate=0.2),

            ResidualBlock(640, 768, 25, dropout_rate=0.3),

            ConvBlock(768, 896, 29, dropout_rate=0.4, dilation=2),

            ConvBlock(896, 1024, 1, dropout_rate=0.4),

            ConvBlock(1024, len(self.vocab), 1)
        )

        self.log_softmax = InferenceBatchSoftmax()


    def forward(self, x, lengths=None):
        """
        x : (batch_size, num_feats, seq_len)
        """
        x = self.layers(x)
        x =  self.log_softmax(x, dim=1)
        if lengths is not None:
            return x, lengths / self.stride # stride reduces seq len
        else:
            return x

    def valid_forward(self, x, lengths=None):
        x = self.layers(x)
        # don't do log_softmax as we need CTC loss from valid batch without logsoftmax
        if lengths is not None:
            return x, lengths / self.stride # stride reduces seq len
        else:
            return x

class JasperSmall(nn.Module):

    def __init__(self, input_dim, embedding_dim, num_classes, inplanes=256, stride=2, margin=0.2, scale=30.0, easy_margin=False):
        super(JasperSmall, self).__init__()

        self.prepooling_layers = nn.Sequential(
            ConvBlock(input_dim, inplanes, 11, stride=stride, dropout_rate=0.2),

            ResidualBlock(inplanes, inplanes, 11, dropout_rate=0.2),

            ResidualBlock(inplanes, inplanes+inplanes//2, 13, dropout_rate=0.2),

            ResidualBlock(inplanes+inplanes//2, 2*inplanes, 17, dropout_rate=0.2),

            ConvBlock(2*inplanes, 3*inplanes, 19, dropout_rate=0.3, dilation=2),

            ConvBlock(3*inplanes, 3*inplanes+inplanes//2, 1, dropout_rate=0.4),

            ConvBlock(3*inplanes+inplanes//2, 1024, 1)
        )


        # pooling defined below                                                                                                                                                                                                                                                                               

        # embedding                                                                                                                                                                                                                                                                                           
        self.embedding = nn.Linear(2048, embedding_dim)
        nn.init.kaiming_normal_(self.embedding.weight)

        self.arc_logit = ArcMarginModel(num_classes, embedding_dim, easy_margin, margin, scale)
        self.base_margin = margin


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


    def forward(self, x, labels=torch.arange(2), epoch=1):
        # Compute embeddings                                                                                                                                                                                                                                                                                                   
        x = self.extract_embedding(x)
        if epoch < 2:
            self.arc_logit.set_margin(0.0)                                                                                                                                                                                                                                                                     
        else:
            self.arc_logit.set_margin(self.base_margin)

        x = self.arc_logit(x, labels)

        return x

# RESNET 1D

def conv3_1D(in_planes, out_planes, stride=1):
    """3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1_1D(in_planes, out_planes, stride=1):
    """1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock_1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_1D, self).__init__()
                                        
        self.conv1 = conv3_1D(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3_1D(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class angular_ResNet_1D(nn.Module):

    def __init__(self, input_dim, num_classes, block=BasicBlock_1D, layers=[3, 4, 6, 3],  embedding_dim=256, inplanes=32, zero_init_residual=True, margin=0.2, scale=30.0, easy_margin=False):
        super(angular_ResNet_1D, self).__init__()
        
        self._norm_layer = nn.BatchNorm1d
        self.inplanes = inplanes
                            
        self.conv1 = nn.Conv1d(input_dim, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, 2*inplanes, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4*inplanes, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8*inplanes, layers[3], stride=2)
                
        # Use mean_std_dev_pooling
        
        # Compute embedding
        self.embedding = nn.Linear(16*inplanes, embedding_dim)

        self.arc_logit = ArcMarginModel(num_classes, embedding_dim, easy_margin, margin, scale)
        self.base_margin = margin

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():                                
                if isinstance(m, BasicBlock_1D):
                    nn.init.constant_(m.bn2.weight, 0)  

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
                
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1_1D(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def mean_std_pooling(self, x, eps=1e-9):
        m = torch.mean(x, dim=2)
        s = torch.sqrt(torch.mean((x - m.unsqueeze(2))**2, dim=2) + eps)
        x = torch.cat([m, s], dim=1)
        return x

    def extract_embedding(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)        

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = torch.flatten(x, 1, 2)
        x = self.mean_std_pooling(x)        
        x = self.embedding(x)

        return x
    
    
    def forward(self, x, labels=torch.arange(2), epoch=1):
        # Compute embeddings                                                                                                                                                                                                                                                                                                   
        x = self.extract_embedding(x)
        if epoch < 2:
            self.arc_logit.set_margin(0.0)                                                                                                                                                                                                                                                                     
        else:
            self.arc_logit.set_margin(self.base_margin)

        x = self.arc_logit(x, labels)

        return x

        
    
#### Explore 1D REsnet
class explore_angular_ResNet_1D(nn.Module):

    def __init__(self, input_dim, num_classes, block=BasicBlock_1D, layers=10,  embedding_dim=256, inplanes=1024, zero_init_residual=True, margin=0.2, scale=30.0, easy_margin=False):
        super(explore_angular_ResNet_1D, self).__init__()

        self._norm_layer = nn.BatchNorm1d
        self.inplanes = inplanes

        self.conv1 = nn.Conv1d(input_dim, self.inplanes, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, inplanes, layers)
                                                                                                                        

        # Use mean_std_dev_pooling                                                                                                                                                                              

        # Compute embedding                                                                                                                                                                                     
        self.embedding = nn.Linear(2*inplanes, embedding_dim)

        self.arc_logit = ArcMarginModel(num_classes, embedding_dim, easy_margin, margin, scale)
        self.base_margin = margin

        # Initialize                                                                                                                                                                                            
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock_1D):
                    nn.init.constant_(m.bn2.weight, 0)
                    
    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1_1D(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def mean_std_pooling(self, x, eps=1e-9):
        m = torch.mean(x, dim=2)
        s = torch.sqrt(torch.mean((x - m.unsqueeze(2))**2, dim=2) + eps)
        x = torch.cat([m, s], dim=1)
        return x

    def extract_embedding(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)

        #x = torch.flatten(x, 1, 2)                                                                                                                                                                             
        x = self.mean_std_pooling(x)
        x = self.embedding(x)

        return x


    def forward(self, x, labels=torch.arange(2), epoch=1):
        # Compute embeddings                                                                                                                                                                                   \
                                                                                                                                                                                                                
        x = self.extract_embedding(x)
        if epoch < 2:
            self.arc_logit.set_margin(0.0)                                                                                                                                                                     \

        else:
            self.arc_logit.set_margin(self.base_margin)

        x = self.arc_logit(x, labels)

        return x

    
    
    
