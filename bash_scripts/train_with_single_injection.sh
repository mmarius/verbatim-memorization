#!/usr/bin/env bash

# Restrict visible GPUs
export CUDA_VISIBLE_DEVICES=$3 

# Update default cache dir of huggingface transformers and datasets
export HF_HOME=/mnt/research/scratch/mmosba/models
export HF_DATASETS_CACHE=/mnt/research/scratch/mmosba/datasets

# Setup wandb
export WANDB_API_KEY= # DO NOT COMMIT THIS!!!
export WANDB_ENTITY=siva-reddy-mila-org
export WANDB_PROJECT=verbatim-memorization
export WANDB_DIR=/mnt/research/scratch/mmosba/verbatim-memorization/wandb-logs
export WANDB_RUN_GROUP="single-sequence-injection"

PROJECT_DIR=/home/nlp/users/mmosba/projects/verbatim-memorization

# Specify arguments
MODEL_NAME_OR_PATH=pythia-160m-deduped-step80000
# MODEL_NAME_OR_PATH=pythia-410m-deduped-step80000
# MODEL_NAME_OR_PATH=pythia-1.4b-deduped-step80000
# MODEL_NAME_OR_PATH=pythia-6.9b-deduped-step80000 # TODO(mm): OOM on McGill cluster

# PILE_DATA=/mnt/research/scratch/mmosba/pythia-data/batches/input_ids-from10000-to11000.npy # 1001 batches of batch size 1024
# PILE_DATA=/mnt/research/scratch/mmosba/pythia-data/batches/input_ids-from40000-to41000.npy # 1001 batches of batch size 1024
# PILE_DATA=/mnt/research/scratch/mmosba/pythia-data/batches/input_ids-from80000-to81000.npy # 1001 batches of batch size 1024
PILE_DATA=/mnt/research/scratch/mmosba/pythia-data/batches/input_ids-from100000-to101000.npy # 1001 batches of batch size 1024

INJECTION_DATA_PATH=$PROJECT_DIR/data/pythia-160m-deduped-step80000_256_256_injection-ppl.json # sorted by perplexity (increasing)
# INJECTION_DATA_PATH=$PROJECT_DIR/data/pythia-410m-deduped-step80000_256_256_injection-ppl.json # sorted by perplexity (increasing)
# INJECTION_DATA_PATH=$PROJECT_DIR/data/pythia-1.4b-deduped-step80000_256_256_injection-ppl.json # sorted by perplexity (increasing)

SEQUENCE_KEY=$1 # provided as an argument, e.g, 0
TRAIN_BATCH_SIZE_PER_GPU=$2 # provided as an argument, e.g, 0
PORT=$4 # provided as an argument, e.g, 0
OUTPUT_DIR=/mnt/research/scratch/mmosba/verbatim-memorization

python $PROJECT_DIR/scripts/train_with_injection_single_shot.py \
    --checkpoint $MODEL_NAME_OR_PATH \
    --pile_data_path $PILE_DATA \
    --injection_data_path $INJECTION_DATA_PATH \
    --sequence_key $SEQUENCE_KEY \
    --window_size 256 \
    --training_batch_size $TRAIN_BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 100 \
    --lr 0.0001 \
    --stop_after_n_steps 1000 \
    --log_every_n_steps 10 \
    --eval_every_n_steps 50 \
    --compute_mem_every_n_steps 50 \
    --save_every_n_steps 100 \
    --output_dir $OUTPUT_DIR \
    --port $PORT