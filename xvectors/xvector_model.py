from __future__ import print_function
from collections import OrderedDict
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from xvectors.plda_lib import PLDA, GaussLinear, GaussQuadratic
from xvectors.kaldi_feats_dataset import KaldiFeatsDataset, spkr_split

import json

logger = logging.getLogger(__name__)

# Function to initialize model weights
def init_weight(model):

    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            logger.info("Initializing %s with kaiming normal" % str(m))
            nn.init.kaiming_normal_(m.weight, a=0.01) # default negative slope of LeakyReLU
        elif isinstance(m, nn.BatchNorm1d):
            if m.affine:
                logger.info("Initializing %s with constant (1,. 0)" % str(m))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# Function for mean/std pooling
def mean_std_pooling(x, eps=1e-9, T0=0.0):

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
    w = x.new_zeros((N,1))
    w[:] = T / float(T + T0)
    return x, w

# Function for ETDNN convolution block
def conv_block(index, in_channels, out_channels, kernel_size, dilation, padding, bn_momentum=0.1):

    layers = [('conv%d' % index, nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)),
              ('relu%d' % index, nn.LeakyReLU(inplace=True)),
              ('bn%d' % index, nn.BatchNorm1d(out_channels, momentum=bn_momentum, affine=False)),
              ('linear%d' % index, nn.Conv1d(out_channels, out_channels, kernel_size=1)),
              ('relu%da' % index, nn.LeakyReLU(inplace=True)),
              ('bn%da' % index, nn.BatchNorm1d(out_channels, momentum=bn_momentum, affine=False))]
    return layers

# Function for expansion layer
def expansion_layer(layer_dim, bn_momentum=0.1):

    layers = [('expand_linear', nn.Conv1d(layer_dim, layer_dim*3, kernel_size=1)),
              ('expand_relu', nn.LeakyReLU(inplace=True)),
              ('expand_bn', nn.BatchNorm1d(layer_dim*3, momentum=bn_momentum, affine=False))]
    return layers

# Freeze functions
def freeze_prepooling(model):

    # set model to eval mode and turn off gradients
    model.prepooling_frozen = True
    for param in model.prepooling_layers.parameters():
        param.requires_grad = False

def freeze_embedding(model):

    # Freeze embedding and everything before it
    model.embedding_frozen = True
    freeze_prepooling(model)
    for param in model.embedding.parameters():
        param.requires_grad = False

def train_with_freeze(model):

    # Set training mode except for frozen layers
    model.train()
    if not hasattr(model, 'module'):
        if model.prepooling_frozen:
            model.prepooling_layers.eval()
        if model.embedding_frozen:
            model.embedding.eval()

# Function for ETDNN prepooling
def etdnn_prepool(input_dim, layer_dim, embedding_dim):

    layers = []

    # conv blocks
    layers.extend(conv_block(1, input_dim, layer_dim, 5, 1, 2))
    layers.extend(conv_block(2, layer_dim, layer_dim, 3, 2, 2))
    layers.extend(conv_block(3, layer_dim, layer_dim, 3, 3, 3))
    layers.extend(conv_block(4, layer_dim, layer_dim, 3, 4, 4))

    # expansion layer
    layers.extend(expansion_layer(layer_dim))

    return(nn.Sequential(OrderedDict(layers)))

class Xvector_embed(nn.Module):

    def __init__(self, input_dim, layer_dim, embedding_dim, bn_momentum=0.1, relu_flag=False, conf_flag=False):

        super(Xvector_embed, self).__init__()

        # Prepooling (parallel)
        self.prepooling_layers = etdnn_prepool(input_dim, layer_dim, embedding_dim)
        #self.prepooling_layers = nn.DataParallel(self.prepooling_layers)

        # pooling is a function

        # embedding
        embed_layers = [('embed', nn.Linear(layer_dim*6, embedding_dim))]
        if relu_flag:
            # ReLU in embedding
            embed_layers.extend([('embed_relu', nn.LeakyReLU(inplace=True))])
        #embed_layers.extend([('embed_bn', nn.BatchNorm1d(embedding_dim, momentum=bn_momentum, affine=False))])
        embed_layers.extend([('embed_bn', nn.BatchNorm1d(embedding_dim, momentum=bn_momentum, affine=True))])
        self.embedding = nn.Sequential(OrderedDict(embed_layers))

        if conf_flag:
            # confidence
            conf_layers = [('conf', nn.Linear((layer_dim*6)+1, 1))]
            conf_layers.extend([('conf_sigmoid', nn.Sigmoid())])
            self.conf = nn.Sequential(OrderedDict(conf_layers))

    def forward(self, x):
        x = self.prepooling_layers(x)
        x, w = mean_std_pooling(x)
        if hasattr(self, 'conf'):
            w = self.conf(torch.cat([x, w], dim=1))
        else:
            w = None
        x = self.embedding(x)

        return x, w


class Xvector9s(nn.Module):

    def __init__(self, input_dim, layer_dim, embedding_dim, num_classes, LL='linear', N0=9, fixed_N=True,
                 length_norm=False, r=0.9, enroll_type='Bayes', loo_flag=True, bn_momentum=0.1, log_norm=True,
                 conf_flag=False, resnet_flag=True, embed_relu_flag=False):
        """
        Notes:
             1 - loo_flag should be a setter/getter, not in the constructor as it is set
                 as training progresses
        """

        super(Xvector9s, self).__init__()
        self.LL = LL
        self.enroll_type = enroll_type
        self.r = r
        self.loo_flag = loo_flag
        self.prepooling_frozen = False
        self.embedding_frozen = False

        if LL == 'xvec':
            # X-vector has ReLU not length-norm
            embed_relu_flag = True
            length_norm = False

        update_plda = False
        update_scale = False
        if LL == 'Gauss' or LL == 'None':
            if length_norm:
                # Update scale factor if length-norm and generative Gaussian
                update_scale = True
            if enroll_type != 'ML':
                # Bayesian or MAP enrollment: PLDA is relevant
                update_plda = True
        self.update_plda_flag = update_plda

        # embedding
        if resnet_flag:
            self.embed = ResNet_embed(input_dim, block=BasicBlock, layers=[3, 4, 6, 3],  embedding_dim=embedding_dim, inplanes=32, zero_init_residual=True, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None)
        else:
            self.embed = Xvector_embed(input_dim, layer_dim, embedding_dim, bn_momentum, embed_relu_flag, conf_flag)
        self.embed = nn.DataParallel(self.embed)

        # length-norm and PLDA layer
        self.plda = PLDA(embedding_dim, num_classes, length_norm, update_scale, update_plda, log_norm=log_norm)

        # output 
        if self.LL == 'None':
            self.output = None

        elif self.LL == 'Gauss':
            # Gaussian generative: can be linear or quadratic depending upon enroll_type
            if enroll_type == 'ML':
                self.output = GaussLinear(embedding_dim, num_classes, N0, fixed_N)
            else:                
                self.output = GaussQuadratic(embedding_dim, num_classes, N0, fixed_N, r, enroll_type)

        elif self.LL == 'Gauss_discr':
            # Gaussian discriminative means
            self.output = GaussLinear(embedding_dim, num_classes, N0, fixed_N, discr_mean=True)

        elif self.LL == 'linear' or self.LL == 'xvec':
            # Linear layer
            self.output = nn.Linear(embedding_dim, num_classes)

        else:
            raise ValueError("Invalid log-likelihood output type %s." % self.LL)

        # initialize
        init_weight(self)

    def forward(self, x, labels=None, embedding_only=False):

        # Compute embeddings 
        x, w = self.embed(x)

        # Length norm and PLDA
        y, z = self.plda(x)

        # Training and not leave-one-out: update parameters
        if labels is not None and self.training and not self.loo_flag:
            self.update_params(x, y, z, labels)

        # Compute output class log likelihoods
        if embedding_only or self.output is None:
            LL = None
        elif self.LL == 'Gauss_discr' or self.LL == 'linear' or self.enroll_type == 'ML':
            # Discriminative outer layer: take pre-LDA output
            LL = self.output(y)
            if w is not None:
                LL = LL / w
        else:
            # Gaussian after PLDA
            LL = self.output(z, w, self.plda.Ulda, self.plda.d_wc, self.plda.d_ac)

        # Training and leave-one-out: update parameters
        if labels is not None and self.training and self.loo_flag:
            self.update_params(x, y, z, labels)

        return x, y, z, LL, w

    def update_params(self, x, y, z, labels):

        if self.update_plda_flag:
            self.plda.update_plda(y, labels)
        if self.LL == 'Gauss':
            self.output.update_params(y, labels)
        return

    def freeze_prepooling(self):

        model = self.embed
        if hasattr(model, 'module'):
            model = model.module
        freeze_prepooling(model)

        return


### Resnet embedding functions
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

class ResNet_embed(nn.Module):

    def __init__(self, input_dim, block=BasicBlock, layers=[3, 4, 6, 3],  embedding_dim=256, inplanes=32, zero_init_residual=True, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet_embed, self).__init__()
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
        
        # embedding
        embed_layers = [('embed', nn.Linear((input_dim//4) * 8*inplanes, embedding_dim))]
        if 0:
            # ReLU in embedding
            embed_layers.extend([('embed_relu', nn.LeakyReLU(inplace=True))])
        embed_layers.extend([('embed_bn', nn.BatchNorm1d(embedding_dim, affine=True))])
        self.embedding = nn.Sequential(OrderedDict(embed_layers))

        # Hard-coded initialization
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

    def mean_std_pooling(self, x, eps=1e-9, std_flag=True):
        m = torch.mean(x, dim=2)
        if std_flag:
            s = torch.sqrt(torch.mean((x - m.unsqueeze(2))**2, dim=2) + eps)
            x = torch.cat([m, s], dim=1)
        else:
            x = torch.cat([m, 0*m], dim=1)
        return x

    def extract_embedding(self, x):
        x = x.unsqueeze(1)
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
        w = None
        return x, w
