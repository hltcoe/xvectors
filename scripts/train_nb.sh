#!/bin/bash
. /etc/profile.d/modules.sh
gpus=$(get_cuda_visible_devices.py --num-gpus 1)
export CUDA_VISIBLE_DEVICES=$gpus
#
source deactivate
conda activate xvec # TODO: replace with your conda environment name

#
env | sort
nvidia-smi
ulimit -a
#

# Directories
#FEATS='feats_preprocess'
FEATS='feats_t'
#MODEL_DIR='./models'
MODEL_DIR="/expscratch/kkarra/xvec_retrain/nb_test"
mkdir -p $MODEL_DIR
TRAIN_DIR='/expscratch/amccree/data/pytorch/fbank_64/'

# make sure the files needed for training exist!
if [[ ! -f $TRAIN_DIR/$FEATS.scp || ! -f $TRAIN_DIR/utt2spk ]]; then
  echo "${TRAIN_DIR}/${FEATS}.scp or ${TRAIN_DIR}/utt2spk do not exist!"
fi

# Flag to copy data to scratch disk first (takes 20 min)
DATA_COPY=0

# Options
MODEL_OPTS="--feature-dim=64 --embedding-dim=128 --layer-dim=768 --length_norm"
# NOTE:
#   1. To refine/optimize the DNN for a specific frame-length, modify the min-frames and max-frames
#      For example: to refine/optime to 2s segment length, set --min-frames=200 --max-frames=200
#   2. For general training, it may be more advantageous to set a wider range for --min-frames and
#       --max-frames, to allow the DNN to be trained for a more general purpose scenario
#       For general training, we set --min-frames=100 and --max-frames=250
FRAME_OPTS="--random-frame-size --min-frames=100 --max-frames=250"

TRAIN_OPTS="--LLtype=Gauss --train_cost=CE --batch-size=512"

ENROLL_OPTS="--enroll_R=0.9 --enroll_type=Bayes"

VALID_OPTS=""
#VALID_OPTS=" --valid_only "  # NOTE: if this is set, only run validation of a trained model

SCHEDULE_OPTS="--epochs=600 --init_up 30"
OPTIM_OPTS="--optimizer=sgd --momentum=0.9 --learning-rate=0.1 --weight-decay=1e-4 $SCHEDULE_OPTS"

# Look for initial model for refinement
INIT_MOD=$(/bin/ls -t $MODEL_DIR/model_init.pth)
if [ ! -z "$INIT_MOD" ]; then
  echo "Initial model $INIT_MOD found, refinement training only."
  INIT_OPTS="--load_model=$INIT_MOD --freeze_prepool"
  OPTIM_OPTS="--optimizer=sgd --momentum=0.9 --learning-rate=0.01 --weight-decay=1e-4 --epochs=60 --init_up 3"
fi

# Look for best model or latest checkpoint
BEST_MOD=$(/bin/ls -t $MODEL_DIR/best-model.pth)
if [ ! -z "$BEST_MOD" ]; then
  echo "Best model $BEST_MOD found, resuming from there."
  INIT_OPTS="--resume-checkpoint=$BEST_MOD"
else
  # Look for latest checkpoint
  CHKPT=$(/bin/ls -t $MODEL_DIR/ch*.pth | head -1)
  if [ ! -z "$CHKPT" ]; then
    echo "Latest checkpoint $CHKPT found, resuming from there."
    INIT_OPTS="--resume-checkpoint=$CHKPT"
  fi
fi

# Copy training feats to tmpdir if available
if [ $DATA_COPY -eq 1 -a -n "$TMPDIR" ]; then
  dfree=$(df -k --output=avail $TMPDIR | tail -1)
  fsize=$(du -k $TRAIN_DIR/$FEATS.ark | cut -f 1)
  echo "$TMPDIR free $dfree size needed is $fsize"
  if [ "$dfree" -lt $((2 * $fsize)) ]; then
    echo "Not enough space on $TMPDIR to copy files from $TRAIN_DIR."
  else
    echo "Copying files to $TMPDIR from $TRAIN_DIR..."
    cp $TRAIN_DIR/utt2spk $TMPDIR
    cp $TRAIN_DIR/$FEATS.scp $TMPDIR
    python search_replace.py $TRAIN_DIR $TMPDIR $TMPDIR/$FEATS.scp $TMPDIR/$FEATS.scp
    echo "Copying archive to $TMPDIR from $TRAIN_DIR..."
    cp $TRAIN_DIR/$FEATS.ark $TMPDIR
    echo "Done."
    TRAIN_DIR=$TMPDIR
  fi
fi

set -x
python train_from_feats.py \
  ${MODEL_OPTS} ${FRAME_OPTS} ${TRAIN_OPTS} ${ENROLL_OPTS} ${VALID_OPTS} ${OPTIM_OPTS} ${INIT_OPTS} \
  --num-workers=8 \
  --log-interval=1000 \
  --train-portion=0.9 \
  --checkpoint-dir=$MODEL_DIR \
  ${TRAIN_DIR}/${FEATS}.scp ${TRAIN_DIR}/utt2spk
