import argparse
import collections
import gc
import json
import numpy as np
import os
import re
import sys

MEM_LIB_DIR = "/home/nlp/users/mmosba/projects/verbatim-memorization/src"
sys.path.append(MEM_LIB_DIR)

from distributed_train import run_worker
import torch
import torch.multiprocessing as mp


def _check_args(args):
    assert args.epochs == 1, "Only one epoch is supported for now."


def main(args):
    world_size = torch.cuda.device_count()  # number of GPUs
    ckpt_name = args.checkpoint

    pile_data = np.load(args.pile_data_path, "r")
    print(f"Load Pile data with shape {pile_data.shape}")
    print(
        f"This corresponds to {pile_data.shape[0] // 1024} steps in the original training (with bsz=1024)"
    )
    print(
        f"And to {pile_data.shape[0] // (args.training_batch_size * world_size)} steps in the new training (with bsz={args.training_batch_size * world_size})"
    )
    
    # load the injection data
    inject_data = json.load(open(args.injection_data_path))

    total_num_occur = args.number_of_occurrences
    inject_every_n = (
        args.inject_every_n_steps
    )  # set this to a big number so that we only inject the sequence once
    window_size = args.window_size
    init_lr = args.lr

    # create an empty directory for the model checkpoint and other log data
    os.makedirs(args.output_dir, exist_ok=True)

    training_batch_size = (
        args.training_batch_size
    )  # actual batch size is batch_size * world_size
    eval_batch_size = args.eval_batch_size

    # get the sequence to be injected
    sequence = inject_data[args.sequence_key]

    print(f"sequence_key={args.sequence_key}, inject_every_n={inject_every_n}")
    print("injecting sequence:")
    print(sequence)

    inject_data = {0: sequence}  # inject a single sequence

    # create training config
    config = {
        "port": args.port,
        "inject_every_n": inject_every_n,
        "total_number_inject": total_num_occur,
        "pile_data_path": args.pile_data_path,
        "inject_data": inject_data,
        "injection_data_path": args.injection_data_path,
        "sequence_key": args.sequence_key,
        # "transformation_type": group_to_inject_data[group + "_transform"],
        "training_batch_size": training_batch_size,
        "eval_batch_size": eval_batch_size,
        # "training_sample_range": [
        #     0,
        #     1000 * 1024,
        # ],  # TODO(mm): where are these coming from? We will just use all of the loaded pile data for training
        # "eval_sample_range": [
        #     1000 * 1024,
        #     1024 * 1024,
        # ],  # TODO(mm): where are these coming from? Let's use a different validation dataset
        "window_size": window_size,
        "base_model": ckpt_name,
        "init_lr": init_lr,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "log_dir": args.output_dir,
        "hf_cache_dir": args.hf_cache_dir,
        "data": pile_data,
        "stop_after_n_steps": args.stop_after_n_steps,  
        "epochs": args.epochs,
        "log_every_n_steps": args.log_every_n_steps,
        "save_every_n_steps": args.save_every_n_steps,
        "run_eval": args.run_eval,
        "eval_every_n_steps": args.eval_every_n_steps,
        "compute_mem_every_n_steps": args.compute_mem_every_n_steps,
        "save_final_checkpoint": args.save_final_checkpoint,
    }

    # start training
    mp.spawn(
        run_worker,
        args=(
            world_size,
            config,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pythia-160m-deduped-step80000",
        help="Path to or name of the model to use",
    )
    parser.add_argument(
        "--pile_data_path", type=str, default="Path to the Pile indices file."
    )
    parser.add_argument(
        "--injection_data_path",
        type=str,
        default="File that contains the injection sequences. This has to be a .json file.",
    )
    # parser.add_argument(
    #     "--inject_sequence_keys",
    #     nargs="+",
    #     default=[],
    #     help="Names of datasets from which to be injected sequences are sampled.",
    # )
    parser.add_argument(
        "--sequence_key",
        type=str,
        default="0",
        help="Id/key of the sequence to be injected.",
    )
    parser.add_argument(
        "--number_of_occurrences",
        type=int,
        default=1,
        help="How many times to inject the sequence.",
    )
    parser.add_argument(
        "--inject_every_n_steps",
        type=int,
        default=1_000_000,
        help="Inject the sequence every n steps.",
    )
    parser.add_argument(
        "--training_batch_size",
        type=int,
        default=16,
        help="Per GPU batch size during training. ",
    )    
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Whether to run evaluation during training.",
    )    
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=100,
        help="Per GPU batch size during evaluation.",
    )
    parser.add_argument(
        "--compute_mem_every_n_steps",
        type=int,
        default=100,
        help="Estimate memorization every n steps.",
    )
    parser.add_argument(
        "--eval_every_n_steps",
        type=int,
        default=100,
        help="Run evaluation every n steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of batches to accumulate before performing an update.",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Log training metrics every n steps.",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=1000,
        help="Save a training checkpoint every n steps.",
    )
    parser.add_argument(
        "--stop_after_n_steps",
        type=int,
        default=200,
        help="Stop training after n steps.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train for.",
    )
    parser.add_argument("--window_size", type=int, default=224, help="Context size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Where to save model checkpoints and log files.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default="/mnt/research/scratch/mmosba/hf-cache-dir",
        help="Where to save hf artifacts (datasets and models).",
    )
    parser.add_argument(
        "--save_final_checkpoint",
        action="store_true",
        help="Whether to save the final checkpoint after training.",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="12358",
        help="Which port to use for DDP communication.",
    )   
    args = parser.parse_args()
    # run some assertions on the arguments
    _check_args(args)
    # run the main function
    main(args)
