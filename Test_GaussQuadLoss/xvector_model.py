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

class TransposeLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(TransposeLinear, self).__init__(in_features, out_features, True)

    def forward(self, input):
        return F.linear(input.transpose(2, 1), self.weight, self.bias).transpose(2, 1)

class Xvector9s(nn.Module):
    # LL can be linear or Gauss
    def __init__(self, input_dim, layer_dim, embedding_dim, num_classes, T0=0.0, length_norm=False):
        super(Xvector9s, self).__init__()
        self.T0 = T0 # duration model
        if T0:
            logger.info("Duration modeling with T0=%.2f",T0)
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
            if isinstance(m, nn.Linear):
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

    def forward(self, x):

        # Compute embeddings 
        x, w = self.extract_embedding(x)

        # Length norm and PLDA
        y, z = self.plda(x)

        return x, y, z, w

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
        #z = y
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
        plda_ds = 1
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
                    # Direct
                    eval, self.Ulda.data = torch.symeig(cov_ac, eigenvectors=True)
                else:
                    # Incremental update from last time: not working
                    S = torch.mm(torch.mm(torch.t(self.Ulda),cov_ac),self.Ulda)
                    eval, U2 = torch.symeig(S, eigenvectors=True)
                    U = torch.mm(self.Ulda,U2)
                    # keep Ulda orthogonal
                    I = torch.eye(cov_ac.shape[0],device=cov_ac.device)
                    self.Ulda.data = torch.mm(U, I + 0.5*(I-torch.mm(torch.t(U),U)))

                self.d_ac.data = torch.diag(torch.mm(torch.mm(torch.t(self.Ulda),cov_ac),self.Ulda))


# Function to update running sums and counts for classes
def update_counts(x, labels, sums, counts, N0):

    N = x.shape[0]
    l = (1.0/N0)
    for n in range(N):
        m = labels[n]
        if counts[m] < (N0-0.5):
            # Running sum at first
            sums[m,:] += x[n,:]
            counts[m] += 1

        else:
            # Recursive sum and count for forgetting
            sums[m,:] = (N0/counts[m])*(1-l)*sums[m,:] + x[n,:]
            counts[m] = N0
        if m == 100:
            logger.info(" spkr100 count %.2f", counts[m])
            print(sums[m,0:2])
       
class GaussQuadLoss(nn.Module):
    def __init__(self, d=None, M=None, plda=None, N0=9, fixed_N=True, r=0.9, enroll_type='Bayes', ge2e=False):
        super(GaussQuadLoss, self).__init__()

        self.plda = plda
        self.N0 = N0     # running sum size for mean updates (enrollment)
        self.fixed_N = fixed_N # fixed or random N0 per batch
        self.r = r # cut correlation
        self.enroll_type = enroll_type
        self.discr_mean = (enroll_type == 'discr')
        self.ge2e = ge2e
        if not ge2e:
            if self.discr_mean:
                # Discriminative training of means 
                self.means = nn.Parameter(torch.zeros(M,d), requires_grad=True)
            else:
                # Generative means
                self.means = nn.Parameter(torch.zeros(M,d), requires_grad=False)
            self.cov = nn.Parameter(torch.zeros(M,d), requires_grad=False)
            self.register_buffer('sums', torch.zeros(M,d))
            self.register_buffer('counts', torch.zeros(M,))
        self.N_dict = {}

    def forward(self, y, z, w, labels, eval_mode=False):

        if self.ge2e:
            # GE2E: call specialized loss function using local models in minibatch
            loss, acc = GaussLoss(z, w, labels, self.plda.d_wc, self.plda.d_ac, self.r, self.enroll_type, self.N_dict)
            return loss,acc

        if not (self.discr_mean or eval_mode):

            # Generative model enrollment from stats
            with torch.no_grad():
                self.means.data, self.cov.data = gmm_adapt(self.counts, torch.mm(self.sums,self.plda.Ulda), self.plda.d_wc, self.plda.d_ac, self.r, self.enroll_type, self.N_dict)

               # Update running stats
                if self.fixed_N:
                    N0 = self.N0
                else:
                    N0 = random.randint(1,(2*self.N0)-1)
                update_counts(y, labels, self.sums, self.counts, N0)

        # Compute log-likelihoods of all models on minibatch
        cov_test = 1.0
        LL = gmm_score(z, self.means, self.cov+cov_test)

        # Return normalized cross entropy cost
        loss = NCE_loss(LL, labels)
        acc = accuracy(LL,labels)
        return loss,acc

    def mean_loss(self):
        means = self.sums / self.counts[:,None]
        loss = torch.mean((self.means-means)**2)
        return loss

# Wrapper function for normalized cross entropy loss and accuracy
#  can do outer layer or gaussian minibatch means
def ComputeLoss(x, y, output, w, labels, loss_type='CE', model=None):

    if loss_type == 'CE':
        loss = NCE_loss(output, labels)
        acc = accuracy(output, labels)
    elif loss_type == 'GaussLoss':
        loss, acc = GaussLoss(y, w, labels, cov_ac=model.PLDA.d_ac)
    else:
        raise ValueError("Invalid loss type %s." % loss_type)

    if 0 and model is not None:
        # add in penalty on centers
        Cl = 1e-2
        loss += Cl*model.PLDA.center_loss(x)
        loss += Cl*model.PLDA.norm_loss(x)

        if model.output.discr_mean:
            # penalty on means vs. ML
            loss += Cl*model.output.mean_loss()

    return loss, acc

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
        loss = (1.0/C0)*F.cross_entropy(LL, labels)
    else:
        loss = torch.tensor(0.0)
    return loss

# Gaussian diarization loss in minibatch
# Compute Gaussian cost across minibatch of samples vs. average
def GaussLoss(x, w, labels, cov_wc=None, cov_ac=None, r = 0.9, enroll_type='Bayes', N_dict=None, loo_flag=True):

    N = x.shape[0]
    d = x.shape[1]
    classes = list(set(labels.tolist()))
    M = len(classes)
    l2 = labels.clone()
    sums = x.new_zeros((M,d))
    counts = x.new_zeros((M,))
    cov_test = 1.0
    if N_dict is None:
        N_dict = {}

    # Compute stats for classes
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

    # Return normalized cross entropy cost
    loss = NCE_loss(LL, l2, prior)
    acc = accuracy(LL,l2)
    return loss,acc

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
