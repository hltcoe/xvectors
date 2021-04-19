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

def Ulda_update0(cov, U):

    d = cov.shape[0]
    I = torch.eye(d, device=cov.device)
    for iter in range(1):

        Ut = torch.t(U)
        R = I - torch.mm(Ut,U)
        S = torch.mm(torch.mm(Ut,cov),U)
        l = torch.diag(S) / (1.0-torch.diag(R))
        E = 0*I
        delta = 2*(torch.norm(S-torch.diag(l)) + torch.norm(cov)*torch.norm(R))
        d1 = 2*(torch.norm(S-torch.diag(l)))
        d2 = 2*(torch.norm(cov)*torch.norm(R))
        #print("Ulda iter ", iter, delta, d1, d2)
        print("Ulda iter %d %.3f %.3f %.3f" %(iter, delta, d1, d2))
        return U

        if 0 and delta > 0.01:
            print(" delta too large, running full Ulda update")
            S = torch.mm(torch.mm(torch.t(U),cov),U)
            ev,U2 = Jacobi(S)
            U = torch.mm(U,U2)
            # keep orthogonal
            U = torch.mm(U, I + 0.5*(I-torch.mm(torch.t(U),U)))
            return U

        #delta =1e-2
        for i in range(d):
            for j in range(d):
                diff = l[i]-l[j]
                if torch.abs(diff) > delta:
                    E[i,j] = (S[i,j] + l[j]*R[i,j]) / diff
                else:
                    E[i,j] = 0.5*R[i,j]
        U = U + torch.mm(U,E)
        
    return U

def Ulda_update(cov, U):

    #if 1:
    for iter in range(1):
        U = Ulda_update0(cov,U)
        S = torch.mm(torch.mm(torch.t(U),cov),U)
        #ev,U2 = Jacobi(S.numpy())
        #U = torch.mm(U,torch.from_numpy(U2))
        if 1:
            ev,U2 = Jacobi(S)
            U = torch.mm(U,U2)
        else:
            S = cov.clone()
            ev,U = Jacobi(S, U)

        # keep orthogonal
        I = torch.eye(cov.shape[0],device=cov.device)
        U = torch.mm(U, I + 0.5*(I-torch.mm(torch.t(U),U)))

        U = Ulda_update0(cov,U)

    return U

def Jacobi(A, U=None):
    n     = A.shape[0]            # matrix size #columns = #lines
    #maxit = 100                   # maximum number of iterations
    maxit = 30                   # maximum number of iterations
    eps   = 1.0e-15               # accuracy goal
    pi    = np.pi
    if U is None:
        U     = torch.eye(n,device=A.device) # initialize eigenvector

    ndone=0
    for t in range(0,maxit):
        Au = torch.abs(torch.triu(A,diagonal=1)) # abs of upper triangular off-diagonal
        limit = torch.mean(Au)
        Amax = torch.max(Au)
        Amin = torch.min(Au)
        #print("iter, limit ", t, limit, Amin, Amax)
        #for i in range(0,n-1):       # loop over lines of matrix
            #for j in range(i+1,n):  # loop over columns of matrix
                #if (torch.abs(A[i,j]) > limit):      # determine (ij) such that |A(i,j)| larger than average 
                                                 # value of off-diagonal elements
        thresh=0.5*Amax
        #thresh = limit
        lam = 0.01
        thresh = lam*limit+(1-lam)*Amax
        Aind = torch.nonzero(torch.abs(Au).gt(thresh))
        #print(Aind)
        for ind in range(Aind.shape[0]):
            i = Aind[ind][0]
            j = Aind[ind][1]
            if 1:
                if 1:
                    denom = A[i,i] - A[j,j]       # denominator of Eq. (3.61)
                    if (torch.abs(denom) < eps): 
                        phi = pi/4         # Eq. (3.62)
                    else: 
                        phi = 0.5*torch.atan(2.0*A[i,j]/denom)  # Eq. (3.61)
                    si = torch.sin(phi)
                    co = torch.cos(phi)
                    store  = A[i,i+1:j].clone()
                    A[i,i+1:j] = A[i,i+1:j]*co + A[i+1:j,j]*si  # Eq. (3.56) 
                    A[i+1:j,j] = A[i+1:j,j]*co - store *si  # Eq. (3.57) 
                    store  = A[i,j+1:n].clone()
                    A[i,j+1:n] = A[i,j+1:n]*co + A[j,j+1:n]*si  # Eq. (3.56) 
                    A[j,j+1:n] = A[j,j+1:n]*co - store *si  # Eq. (3.57) 
                    store  = A[0:i,i].clone()
                    A[0:i,i] = A[0:i,i]*co + A[0:i,j]*si
                    A[0:i,j] = A[0:i,j]*co - store *si
                    store = A[i,i]
                    A[i,i] = A[i,i]*co*co + 2.0*A[i,j]*co*si +A[j,j]*si*si  # Eq. (3.58)
                    A[j,j] = A[j,j]*co*co - 2.0*A[i,j]*co*si +store *si*si  # Eq. (3.59)
                    A[i,j] = 0.0                                            # Eq. (3.60)
                    store  = U[0:n,j].clone()
                    U[0:n,j] = U[0:n,j]*co - U[0:n,i]*si  # Eq. (3.66)
                    U[0:n,i] = U[0:n,i]*co + store *si  # Eq. (3.67)
                    ndone += 1

        print("ndone " ,ndone)
        if ndone > 1000:
            break

    ev = torch.diag(A)
    return ev,U

