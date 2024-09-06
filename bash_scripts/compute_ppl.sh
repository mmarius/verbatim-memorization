#!/usr/bin/env bash

PROJECT_DIR=/home/nlp/users/mmosba/projects/verbatim-memorization

# Specify arguments
# MODEL_NAME_OR_PATH=EleutherAI/pythia-160m
MODEL_NAME_OR_PATH=EleutherAI/pythia-1.4B
MAX_SEQ_LENGTH=$1
DEVICE=$2
OUTPUT_DIR=$PROJECT_DIR/data

python $PROJECT_DIR/scripts/compute_ppl.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --batch_size 1 \
    --max_seq_length $MAX_SEQ_LENGTH \
    --stride $MAX_SEQ_LENGTH \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE
