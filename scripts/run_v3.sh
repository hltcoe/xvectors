#!/bin/bash -x
. /etc/profile.d/modules.sh
module load cuda90/toolkit/9.0.176
gpus=$(/home/hltcoe/jfarris/tensorroad/get_cuda_visible_devices.py --num-gpus 1)
export CUDA_VISIBLE_DEVICES=$gpus
# randomly wait to prevent random tensorflow issues
SLEEP_TIME=3
#sleep $SLEEP_TIME
export PATH=/opt/anaconda3/bin:$PATH
#
source deactivate
conda info --envs
source activate pytorch-conda
#
env | sort
nvidia-smi
ulimit -a
#

# Point to source code
#CODEBASE="/home/hltcoe/amccree/src/pytorch-xvec"
CODEBASE="/home/hltcoe/amccree/src/pytorch-xvec/Releases/v2"

MODEL_DIR='.'
mkdir -p $MODEL_DIR/log

RESUME=$(/bin/ls -t $MODEL_DIR/*.pth | head -1)
if [ ! -z "$RESUME" ]; then
    resume_opts="--resume-checkpoint=$RESUME"
fi

#    --length_norm \
$CODEBASE/train_from_feats.py \
    --LLtype='linear' \
    --feature-dim=23 \
    --embedding-dim=256 \
    --layer-dim=256 \
    --num-classes=14846 \
    --batch-size=128 \
    --epochs=500 \
    --learning-rate=0.001 \
    --step-size=100 \
    --min-frames=200 \
    --max-frames=400 \
    --random-frame-size \
    --num-workers=12 \
    --step-decay=0.5 \
    --weight-decay=1e-5 \
    --log-interval=200 \
    --optimizer=adam \
    --checkpoint-dir=$MODEL_DIR $resume_opts \
    --valid-feats-scp=/home/hltcoe/jfarris/tensorroad/data/keras_valid/feats.scp \
    --valid-utt2spk=/home/hltcoe/jfarris/tensorroad/data/keras_valid/utt2spk \
    /home/hltcoe/jfarris/tensorroad/data/train_combined/feats.scp /home/hltcoe/jfarris/tensorroad/data/train_combined/utt2spk
