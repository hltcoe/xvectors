# PyTorch xvector training
This repository contains xvector definitions and training recipes using PyTorch.  Two training recipes are included, for narrowband (8kHz) and wideband (16kHz) data.  The repository is organized into two main folders.  The `scripts` folder contains recipes for training xvector extractors.  The `xvectors` folder is the xvectors module, containing model architecture definitions, data loaders, and helper functions for performing inference (xvector extraction) with trained models.

## Usage
We recommend creating an anaconda environment, cloning the repository, and installing xvectors as an importable module to utilize this repository effectively.  The steps are summarized below:

```buildoutcfg
>> conda create -n speech python=3.8
>> conda activate speech
>> git clone https://github.com/hltcoe/xvectors.git
>> cd xvectors
>> pip install -e .
```

## Model Training
Our setup of xvector training requires that audio features be precomputed.  We use filterbank features created using Kaldi, and require that features used for training be accessible through Kaldi file formats.  More specifically, the files needed for training are:

1. `feats.scp`, `feats.ark`
2. `utt2spk`

Please refer to the Kaldi documentation on how to prepare data in this format.

### Training the Narrowband Model
To train a narrowband model, run the `train_nb.sh` bash script located in the `scripts` directory.  The data sources we utilized for narrowband xvector training include:

    1. LDC2004S07
    2. LDC2011S09
    3. LDC2006S44 
    4. LDC2017S06
    5. LDC2011S08
    6. LDC98S75
    7. LDC2011S04
    8. LDC2012S01
    9. LDC2002S06
    10. LDC2001S13
    11. LDC2011S10
    12. LDC2011S01
    13. LDC99S79
    14. LDC2011S05
    15. VoxCeleb1
    16. VoxCeleb2

All data is also augmented with the MUSAN corpus and telephony codecs.

### Training the Wideband Model
To train a wideband model, run the `train_wb.sh` bash script located in the `scripts` directory.  The data sources we utilized for wideband xvector training include:

    1. VoxCeleb1
    2. VoxCeleb2
    3. LDC2013S03

All data is also augmented with the MUSAN corpus.

## Citations
Snyder, David, et al. "X-vectors: Robust dnn embeddings for speaker recognition." 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018.

## Contact
If you have any comments or questions, please create a GitHub issue.