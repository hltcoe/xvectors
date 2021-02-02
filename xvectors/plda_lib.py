from __future__ import print_function
import logging
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from xvectors.utils import accuracy

logger = logging.getLogger(__name__)

# length-norm and PLDA layer
class plda(nn.Module):
    def __init__(self, embedding_dim, num_classes, length_norm=True, update_scale=True, update_plda=True, N0=30, center_l=1e-4, discr_ac=False, log_norm=True):

        super(plda, self).__init__()

        self.N0 = N0     # running sum size for model mean updates in PLDA
        self.length_norm_flag = length_norm
        self.center = nn.Parameter(torch.zeros(embedding_dim), requires_grad=False)
        self.center_l = center_l
        self.d_wc = nn.Parameter(torch.ones(embedding_dim), requires_grad=False)
        if discr_ac:
            # Discriminative training of across-class diagonal covariance, assume Ulda is fixed.
            self.d_ac = nn.Parameter(torch.ones(embedding_dim), requires_grad=True)
            update_plda = False # not correct
        else:
            # Generative across-class covariance
            self.d_ac = nn.Parameter(torch.ones(embedding_dim), requires_grad=False)
        self.Ulda = nn.Parameter(torch.eye(embedding_dim), requires_grad=False)

        self.log_norm = log_norm
        log_norm_scale_init = 0.5*torch.log(torch.tensor(float(embedding_dim)))
        logger.info("Initializing length_norm scaling with log sqrt(dimension)")
        if length_norm and update_scale:
            if log_norm:
                self.log_norm_scale = nn.Parameter(log_norm_scale_init, requires_grad=True)
            else:
                self.norm_scale = nn.Parameter(torch.exp(log_norm_scale_init), requires_grad=True)
        else:
            if log_norm:
                self.log_norm_scale = nn.Parameter(log_norm_scale_init, requires_grad=False)
            else:
                self.norm_scale = nn.Parameter(torch.exp(log_norm_scale_init), requires_grad=False)
        if update_plda:
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

        if self.log_norm:
            self.log_norm_scale.data = torch.clamp(self.log_norm_scale,min=0.0)
            norm_scale = (torch.exp(self.log_norm_scale))
        else:
            self.norm_scale.data = torch.clamp(self.norm_scale,min=1.0)
            norm_scale = self.norm_scale
        y = norm_scale*F.normalize(x, p=2, dim=1)
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
        #plda_ds = 0
        plda_ds = 10
        with torch.no_grad():

            self.sums, self.counts = update_counts(x, labels, self.sums, self.counts, self.N0)
            self.plda_cnt -= 1
            if self.counts.min() >= self.N0-1 and self.plda_cnt <= 0:
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
                self.d_ac.data = torch.clamp(self.d_ac,min=0.0)
 
# Function to update running sums and counts for classes
def update_counts(x1, labels1, sums1, counts1, N0, rand_flag=False):

    # Use cpu numpy
    x = x1.cpu().detach().numpy()    
    labels = labels1.cpu().numpy()
    device=sums1.device
    sums1, counts1 = sums1.to("cpu"), counts1.to("cpu")
    sums = sums1.numpy()    
    counts = counts1.numpy()    

    N = x.shape[0]
    M = counts.shape[0]
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
        if 0 and m == 100:
            logger.info(" spkr100 count %.2f N0 %2d N1 %2d", counts[m], N0, N1)
            logger.info(" spkr101 count %.2f", counts[m+1])

    # Return to device
    sums1, counts1 = sums1.to(device), counts1.to(device)
    return sums1, counts1

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
        #self.register_buffer('counts', torch.zeros(num_classes,))
        self.register_buffer('counts', torch.ones(num_classes,))
        
        # Initialize stats
        logger.info("Initializing GaussLinear stats with xavier_normal.")
        nn.init.xavier_normal_(self.sums)

    def forward(self, input):

        bias = -0.5*((self.means**2).sum(dim=1))
        return F.linear(input, self.means, bias)

    def update_params(self, x, labels):

        # Update running counters and means for Gaussian last layer
        with torch.no_grad():
            self.sums, self.counts = update_counts(x, labels, self.sums, self.counts, self.N0, rand_flag=(not self.fixed_N))

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
    def __init__(self, embedding_dim, num_classes, N0=9, fixed_N=True, r=0.9, enroll_type='Bayes', N_dict={}, OOS=True):
        super(GaussQuadratic, self).__init__()

        self.num_classes = num_classes
        self.N0 = N0     # running sum size for mean updates (enrollment)
        self.fixed_N = fixed_N # fixed or random N0 per batch
        self.r = r # cut correlation
        self.enroll_type = enroll_type
        self.means = nn.Parameter(torch.zeros(num_classes,embedding_dim), requires_grad=False)
        self.cov = nn.Parameter(torch.zeros(num_classes,embedding_dim), requires_grad=False)
        self.register_buffer('sums', torch.zeros(num_classes,embedding_dim))
        #self.register_buffer('counts', torch.zeros(num_classes,))
        self.register_buffer('counts', torch.ones(num_classes,))
        self.N_dict = N_dict
        self.OOS = OOS
        
        # Initialize stats
        logger.info("Initializing GaussQuadratic stats with xavier_normal.")
        nn.init.xavier_normal_(self.sums)

    def forward(self, x, w=None, Ulda=None, d_wc=None, d_ac=None):

        # Update models and compute Gaussian log-likelihoods
        with torch.no_grad():
            self.means.data, self.cov.data = gmm_adapt(self.counts, torch.mm(self.sums,Ulda), d_wc, d_ac, self.r, self.enroll_type, self.N_dict)

        N = x.shape[0]
        M = self.num_classes
        if w is None:
            LL = gmm_score(x, self.means, self.cov+d_wc[None,:])

            # Subtract open set LL
            if self.OOS:
                LL -= gmm_score(x, 0.0, d_ac[None,:]+d_wc[None,:])
        else:
            # confidence weighting
            LL = x.new_zeros((N,M))
            for n in range(N):
                LL[n,:] = gmm_score(x[n,None], self.means, self.cov+w[None,n])
                if self.OOS:
                    LL[n,:] -= gmm_score(x, 0.0, d_ac[None,:]+w[None,n])

        return LL

    def update_params(self, x, labels):

       # Update running counters and means for Gaussian last layer
        with torch.no_grad():
            self.sums, self.counts = update_counts(x, labels, self.sums, self.counts, self.N0, rand_flag=(not self.fixed_N))


# Wrapper function for normalized cross entropy loss and accuracy
#  can do outer layer or gaussian minibatch means
def ComputeLoss(x, y, output, w, labels, loss_type='CE', model=None, boost=0):

    if loss_type == 'CE':
        loss, nloss = NCE_loss(output, labels, boost=boost)
        acc = accuracy(output, labels)
    elif loss_type == 'BCE':
        # binary cross-entropy
        loss, nloss = BCE_loss(output, labels)
        if 1:
            loss2, nloss2 = NCE_loss(output, labels, boost=boost)
            l = 0.1
            nloss = l*nloss + (1-l)*nloss2
            loss = nloss
        acc = accuracy(output, labels)
    elif loss_type == 'GaussLoss':
        loss, nloss, acc = GaussLoss(y, w, labels, loo_flag=model.loo_flag, cov_ac=model.plda.d_ac, enroll_type=model.enroll_type, r=model.r, N_dict=model.N_dict)
    elif loss_type == 'BinLoss':
        loss, nloss, acc = BinLoss(y, w, labels, loo_flag=model.loo_flag, cov_ac=model.plda.d_ac, enroll_type=model.enroll_type, r=model.r, N_dict=model.N_dict)
    else:
        raise ValueError("Invalid loss type %s." % loss_type)

    if 0 and model is not None:
        # add in penalty on centers
        Cl = 1e-2
        loss += Cl*model.plda.center_loss(x)
        loss += Cl*model.plda.norm_loss(x)

        if model.output.discr_mean:
            # penalty on means vs. ML
            loss += Cl*model.output.mean_loss()

    return loss, nloss, acc

# Normalized multiclass cross-entropy
def NCE_loss(LL, labels, prior=None, boost=0):

    M = LL.shape[1]
    if prior is None:
        # No prior: flat over number of classes
        C0 = torch.log(torch.tensor(M,dtype=LL.dtype,device=LL.device))
    else:
        # Prior given
        C0 = -torch.sum(prior*torch.log(prior))

    # Boosting: lower true scores
    if boost:
        y = labels.cpu().numpy()
        for i, yi in enumerate(np.unique(y)):
            indexes = (y==yi)
            LL[indexes,i] -= boost

    if M > 1:
        loss = F.cross_entropy(LL, labels)
        nloss = (1.0/C0)*loss
    else:
        loss = torch.tensor(0.0)
        nloss = loss
    return loss, nloss

# Binary cross-entropy, round-robin T/NT scoring
def BCE_loss(LLR, labels, Pt=None):

    N = LLR.shape[0]
    M = LLR.shape[1]
    if Pt is None:
        Pt = 1.0/M

    # post target = Pt*LR / (Pt*LR+1-Pt)
    # post nontarget = (1-Pt)/(Pt*LR+1-Pt)

    # Generate target/nontarget masks
    tar_ind = 0*LLR.ge(float('inf'))
    non_ind = ~tar_ind
    for n in range(N):
        m = labels[n]
        tar_ind[n,m] = 1
        non_ind[n,m] = 0
        
    log_prior = np.log(Pt/(1.0-Pt))
    tar_scores = -torch.log(1+torch.exp(-(torch.masked_select(LLR,tar_ind)+log_prior)))
    non_scores = -torch.log(1+torch.exp(torch.masked_select(LLR,non_ind)+log_prior))

    # Sum log probabilities
    loss = Pt*torch.mean(tar_scores) + (1-Pt)*torch.mean(non_scores)
    C0 = Pt*np.log(Pt) + (1-Pt)*np.log(1-Pt)
    nloss = (1.0/C0)*loss

    return nloss, nloss

# Gaussian diarization loss in minibatch
# Compute Gaussian cost across minibatch of samples vs. average
# Note: w is ignored in this version
def GaussMinibatchLL(x1, w, labels, loo_flag=True, cov_wc1=None, cov_ac1=None, enroll_type='Bayes', r=0.9, N_dict=None, binary=False):

    x = x1.cpu()  
    N = x.shape[0]
    d = x.shape[1]
    classes = list(set(labels.tolist()))
    M = len(classes)
    l2 = labels.clone().cpu()
    sums = x.new_zeros((M,d))
    counts = x.new_zeros((M,))
    if cov_wc1 is None:
        cov_wc = x.new_ones((d,))
    else:
        cov_wc = cov_wc1.cpu() 
    if cov_ac1 is None or len(cov_ac1.shape) > 1:
        cov_ac = x.new_ones((d,))
    else:
        cov_ac = cov_ac1.cpu() 
    cov_test = 1.0
    if N_dict is None:
        N_dict = {}

    # Compute stats for classes
    for m in range(M):
        l2[labels==classes[m]] = m
    sums, counts = update_counts(x, l2, sums, counts, N0=1000, rand_flag=0)

    # Compute models and log-likelihoods
    means, cov = gmm_adapt(counts, sums, cov_wc, cov_ac, r, enroll_type, N_dict)
    LL = gmm_score(x, means, cov+cov_test)

    # Leave one out corrections
    if loo_flag:
        for n in range(N):
            m = classes.index(labels[n])
            if counts[m] > 1:
                mu_model, cov_model = gmm_adapt(counts[m:m+1]-1, sums[m:m+1,:]-x[n,:], cov_wc, cov_ac, r, enroll_type, N_dict)
                LL[n,m] = gmm_score(x[n:n+1,:], mu_model, cov_model+cov_test)

    if binary:
        # Subtract open set LL
        LL -= gmm_score(x, 0.0, cov_wc[None,:]+cov_ac[None,:])
        prior = 1.0/M

    else:
        # Compute and apply prior
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

# Binary T/NT diarization loss in minibatch
def BinLoss(x, w, labels, loo_flag=True, cov_wc=None, cov_ac=None, enroll_type='Bayes', r=0.9, N_dict=None):

    # Return normalized cross entropy cost
    LL, prior, l2 = GaussMinibatchLL(x, w, labels, loo_flag, cov_wc, cov_ac, enroll_type, r, N_dict, binary=True)
    loss, nloss = BCE_loss(LL, l2)
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
            if cnt < 1:
                Neff = cnt
            else:
                Neff = (cnt*(1-r)+2*r) / (1.0+r)

            N_dict[cnt] = 1.0 / Neff
            print("cnt not in dict", cnt, N_dict[cnt])
        
        # Update N_eff
        mask = torch.from_numpy(np.array(icnt==cnt, dtype=np.uint8))
        Nsc[mask] = N_dict[cnt]

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

