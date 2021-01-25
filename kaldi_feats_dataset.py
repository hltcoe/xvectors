import logging
from collections import defaultdict
import random
import copy

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from kaldi_io import read_mat, read_mat_head, read_mat_data


logger = logging.getLogger(__name__)


class KaldiFeatsDataset(Dataset):
    def __init__(self, feats_scp_filename, utt2spk_filename, num_frames=300, cost='CE', enroll_N0=9, utt_per_spk=30, spk2int=None):
        """
        """
        self.num_frames = num_frames
        self.cost = cost
        self.enroll_N0 = enroll_N0
        all_feats_scp = self._read_scp(feats_scp_filename)
        all_utt2spk= self._read_scp(utt2spk_filename)
        self.utts = list(set(all_feats_scp.keys()) & set(all_utt2spk.keys()))

        logger.info("Loaded total of %d utts", len(self.utts))
        self.feats_scp = { k : all_feats_scp[k] for k in self.utts }
        self.utt2spk = { k : all_utt2spk[k] for k in self.utts }

        self.spk2utt = defaultdict(list)
        for utt, spk in sorted(self.utt2spk.items()):
            self.spk2utt[spk].append(utt)

        self.spks = sorted(list(self.spk2utt.keys()))
        # Randomize utterances
        for spk in self.spks:
            random.shuffle(self.spk2utt[spk])
        if spk2int is None:
            self.spk2int = {}
            for i, spk in enumerate(self.spks):
                self.spk2int[spk] = i
        else:
            # check that all speakers are in given spk2int
            M0 = len(self.spks)
            for spk in list(self.spks):
                if spk not in spk2int.keys():
                    self.spks.remove(spk)
                    logger.debug("Removing spk %s not in provided spk2int", spk)
                    #raise ValueError("Missing spk %s from provided spk2int" % spk)
            if len(self.spks) < M0:
                logger.info("Removed %d speakers not in provided spk2int", (M0-len(self.spks)))
            self.spk2int = spk2int

        if utt_per_spk is None:
            # No request for number of utts per spkr: approximate overall data size (rounding error)
            utt_per_spk = len(self.utts) // len(self.spks)
            logger.info("No utt_per_sk requested, set to %d based on datasize", utt_per_spk)
        self.length = utt_per_spk*len(self.spks)
        logger.info("Loaded total of %d speakers, epoch is %d", len(self.spks),self.length)
        logger.info(" using feats scp file %s", feats_scp_filename)

    def speaker_select(self, spkr_list):
        M0 = len(self.spks)
        self.spks = sorted(spkr_list)
        M1 = len(self.spks)
        self.spk2int = {}
        for i, spk in enumerate(self.spks):
            self.spk2int[spk] = i
        self.length = M1*(self.length//M0)
        logger.info("Subselected total of %d speakers, epoch is %d", len(self.spks),self.length)

    def set_Gauss_cost(self):
        self.cost = 'GaussLoss'
        logger.info("Setting cost to GaussLoss")

    def set_Bin_cost(self):
        self.cost = 'BinLoss'
        logger.info("Setting cost to BinLoss")

    def set_frame_length(self, num_frames):
        logger.debug("Setting num_frames to %d", num_frames)
        self.num_frames = num_frames

    def _read_scp(self, filename):
        """Read the content of the text file and store it into lists."""
        scp = {}
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line_num, line in enumerate(lines):
                items = line.strip().split()
                if len(items) != 2:
                    logger.warn('Bad line %d : %s', line_num, line)
                    continue

                scp[items[0]] = items[1]
        return scp

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        spk = self.spks[index[0]]
        utt = self.spk2utt[spk][index[1]]
        fstart = index[2]

        # select utterance from speaker index and read features
        start = None
        fd, N, d, ftype, gmin, grange = read_mat_head(self.feats_scp[utt])
        if N > self.num_frames:
            start = int(fstart*(N - self.num_frames - 1))

        if start is None:
            feats2 = np.ascontiguousarray(read_mat_data(fd, N, d, ftype, 0, gmin, grange, N), dtype=np.float32).T
            feats = np.zeros((d, self.num_frames), dtype=np.float32)
            feats[:,0:N] = feats2
            N2 = min(N,self.num_frames-N)
            feats[:,N:(N+N2)] = feats2[:,0:N2]
            logger.debug("Could not find utt for spk %s that was %d frames, repeat/zero-pad utt %s", spk, self.num_frames, utt)
        else:
            feats = np.ascontiguousarray(read_mat_data(fd, self.num_frames, d, ftype, start, gmin, grange, N), dtype=np.float32).T
            logger.debug("Selected spk %s from %s at frame %d to %d (%d tot)", spk, utt, start, start+self.num_frames, N)
            
        # read label
        y = self.spk2int[spk]

        return (feats, y)


class SpkrSampler(Sampler):
    """Randomize N samples from M speakers

    Arguments:
        data_source (Dataset): kaldi dataset to sample from
    """

    def __init__(self, data_source, reset_flag=False, fixed_N=False):
        self.data_source = data_source
        self.M = len(self.data_source.spks)
        self.N = len(self.data_source) // self.M
        self.cost = self.data_source.cost
        self.fixed_N = fixed_N
        self.enroll_N0 = self.data_source.enroll_N0
        self.reset_flag = reset_flag

        # Initialize index and number of utts per speaker
        self.utt_cnt = np.zeros(self.M, dtype=np.int)
        self.utt_num = np.zeros(self.M, dtype=np.int)
        for i, spk in enumerate(self.data_source.spks):
            self.utt_num[i] = len(self.data_source.spk2utt[spk])

        if self.cost == 'CE' or self.cost == 'BCE':
            logger.info("Sampler generating 1 sample from %d random speakers %d times.", self.M, self.N)
        else:
            if self.fixed_N:
                logger.info("Sampler generating %d samples from %d random speakers with %d per batch.", self.N, self.M, self.enroll_N0+1)
            else:
                logger.info("Sampler generating %d samples from %d random speakers with random average %d per batch.", self.N, self.M, self.enroll_N0+1)

    def __iter__(self):

        ind_list = []

        if self.reset_flag:
            # Reset every epoch for repeatable behavior
            self.utt_cnt = 0*self.utt_cnt

        if self.cost == 'CE' or self.cost == 'BCE':
            # Draw 1 cut from random M speakers N times 
            for n in range(self.N):
                mlist = list(range(self.M))
                random.shuffle(mlist)
                for m in mlist:
                    ind_list.append((m, self.utt_cnt[m], random.random()))
                    self.utt_cnt[m] = (self.utt_cnt[m]+1) % (self.utt_num[m])

        else:
            # Draw fixed or random number of cuts (average N0) from random M speakers
            n0 = 0
            needN = True
            N0 = self.enroll_N0+1
            while n0 < self.N:
                mlist = list(range(self.M))
                random.shuffle(mlist)
                for m in mlist:
                    if self.fixed_N:
                        # Fixed number
                        N = N0
                    else:
                        # Random number
                        if needN:
                            # Need to draw new random N
                            N = random.randint(2,N0)
                        else:
                            # Already drew N, so use complement
                            N = (2*N0) - N
                        needN = not needN

                    # Repeat speaker index N times within batch
                    for n in range(N):
                        ind_list.append((m, self.utt_cnt[m], random.random()))
                        self.utt_cnt[m] = (self.utt_cnt[m]+1) % (self.utt_num[m])
                        n0 += 1

        # Return iterator of list
        return iter(ind_list)

    def __len__(self):
        return len(self.data_source)

def SpkrSplit(trainset, train_split):

    # Randomly split speakers of a training set for testing
    M = len(trainset.spks)
    N = len(trainset) // M
    M1 = int(train_split*M)
    logger.info("Splitting %d speakers into %d and %d.", M, M1, M-M1)
    testset = copy.deepcopy(trainset)
    random.shuffle(trainset.spks)
    logger.info(" Test set:")
    testset.speaker_select(trainset.spks[M1:M])
    logger.info(" Training set:")
    trainset.speaker_select(trainset.spks[0:M1])

    return trainset, testset


