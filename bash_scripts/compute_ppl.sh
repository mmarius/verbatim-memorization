#!/usr/bin/env bash

PROJECT_DIR=/home/nlp/users/mmosba/projects/verbatim-memorization

# Specify arguments
MODEL_NAME_OR_PATH=pythia-160m-deduped-step80000
# MODEL_NAME_OR_PATH=pythia-410m-deduped-step80000
# MODEL_NAME_OR_PATH=pythia-1.4b-deduped-step80000

MAX_SEQ_LENGTH=$1
DEVICE=$2
OUTPUT_DIR=$PROJECT_DIR/data/perplexities
HF_CACHE_DIR=/mnt/research/scratch/mmosba/hf-cache-dir

# Use as follows:
# bash bash_scripts/compute_ppl.sh 256 cuda:4

python $PROJECT_DIR/scripts/compute_ppl.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --batch_size 1 \
    --max_seq_length $MAX_SEQ_LENGTH \
    --stride $MAX_SEQ_LENGTH \
    --output_dir $OUTPUT_DIR \
    --hf_cache_dir $HF_CACHE_DIR \
    --device $DEVICE
