import collections
import gc
import json
import numpy as np
import os
import random

import pandas as pd

from generation_utils import generate_batched
from memorization_utils import compute_per_token_pplx, get_memorized_sequences
from nparray_dataset import NumpyArrayDataset
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from torch.utils.data.distributed import DistributedSampler
import transformers
from transformers import AutoConfig, GPTNeoXForCausalLM, AutoTokenizer
from transformers import get_scheduler, get_linear_schedule_with_warmup


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_optimizer_parameters(optimizer):
    return sum(p.numel() for p in optimizer.param_groups[0]["params"])


def lm_train_step(model, input_batch):
    labels = input_batch["input_ids"].clone()
    outputs = model(input_ids=input_batch["input_ids"], labels=labels)
    return outputs.loss, outputs.logits, labels


def compute_metrics(logits, labels):
    with torch.no_grad():
        # Exclude the last token, which does not have a label.
        pred = torch.argmax(logits[:, :-1], dim=-1)
        # Exclude the first token, which does not have a prediction.
        labels = labels[:, 1:]
        token_acc = torch.masked_select(
            (pred == labels).type(torch.float32), labels != -100
        )
        return {
            "token_accuracy": token_acc.mean().float(),
            "last_token_accuracy": torch.reshape(token_acc, [labels.shape[0], -1])[
                :, -1
            ]
            .mean()
            .float(),
        }


def load_model_and_tokenizer(ckpt_name, cache_dir, device, tokenizer_only=False):
    model_id = ckpt_name.rsplit("-", 1)[0]
    if "pythia" in model_id or "neo" in model_id:
        model_id = "EleutherAI/" + model_id
    elif "opt" in model_id:
        model_id = "facebook/" + model_id
    print("Load checkpoint: %s %s" % (model_id, ckpt_name.split("-")[-1]))
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    tokenizer.pad_token = "<|padding|>"
    tokenizer.padding_side = "left"
    if tokenizer_only:
        return tokenizer
    if "pythia" in model_id:
        model = GPTNeoXForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            revision=ckpt_name.split("-")[-1],
        ).to(device)

    # TODO(mm): not supported for now
    # elif "gpt2" in model_id:
    #     model = GPT2LMHeadModel.from_pretrained(
    #         model_id, low_cpu_mem_usage=True, device_map="auto", cache_dir=cache_dir
    #     )
    # elif "gpt-neo" in model_id:
    #     model = GPTNeoForCausalLM.from_pretrained(
    #         model_id, low_cpu_mem_usage=True, device_map="auto", cache_dir=cache_dir
    #     )
    # elif "opt" in model_id:
    #     model = OPTForCausalLM.from_pretrained(
    #         model_id, low_cpu_mem_usage=True, device_map="auto", cache_dir=cache_dir
    #     )
    else:
        raise NotImplementedError

    return model, tokenizer


def print_with_rank(rank, *arg):
    print(f"[RANK {rank}]", *arg)


def setup_ddp(rank, world_size, port="12355"):
    print(f"Setting up DDP (rank={rank}, world_size={world_size}) ...")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def train_distributed_model(rank, world_size, train_dataloader, val_dataloader, config):
    metrics_logger = collections.defaultdict(list)

    # Construct DDP model
    # print_with_rank(rank, torch.cuda.device_count())
    pretrained_model, tokenizer = load_model_and_tokenizer(
        config["base_model"], config["hf_cache_dir"], rank
    )
    print_with_rank(rank, "#layers=%d" % pretrained_model.config.num_hidden_layers)
    device = pretrained_model.device
    # print_with_rank(rank, device)
    # print_with_rank(
    #     rank, f'CUDA allocated {torch.cuda.memory_allocated() / (1024**3)}')
    # Use DataParallel instead of DDP due to the excessive overhead of DDP, which
    # would only fit a batch size of 8 for a 7B model on a 80G GPU.
    # If you need multi-cluster distributed training, use DDP instead.
    # model = DDP(pretrained_model, device_ids=[rank],
    #             gradient_as_bucket_view=True)
    model = DataParallel(pretrained_model, device_ids=[rank])
    # print_with_rank(
    #     rank, f'CUDA allocated {torch.cuda.memory_allocated() / (1024**3)}')

    num_training_steps = len(
        train_dataloader
    )  # this is already rank specific due to DDP

    # Initialize wandb
    run_name = None
    if rank == 0:
        import wandb

        sequence = tokenizer.decode(
            tokenizer(config["inject_data"][0]).input_ids[: config["window_size"]]
        )  # there is only one sequence

        wandb_config = {
            "world_size": world_size,
            "base_model": config["base_model"],
            "training_batch_size": config["training_batch_size"],
            "eval_batch_size": config["eval_batch_size"],
            "window_size": config["window_size"],
            "pile_data_path": config["pile_data_path"],
            "injection_data_path": config["injection_data_path"],
            "sequence_key": config["sequence_key"],
            "inject_every_n": config["inject_every_n"],
            "total_number_inject": config["total_number_inject"],
            "init_lr": config["init_lr"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "run_eval": config["run_eval"],
            # "training_sample_range": config["training_sample_range"],
            # "eval_sample_range": config["eval_sample_range"],
            "epochs": config["epochs"],
            "total_steps_per_rank": num_training_steps,
            "injection_sequence": sequence,
            "stop_after_n_steps": config["stop_after_n_steps"],
            "log_every_n_steps": config["log_every_n_steps"],
            "save_every_n_steps": config["save_every_n_steps"],
            "eval_every_n_steps": config["eval_every_n_steps"],
            "compute_mem_every_n_steps": config["compute_mem_every_n_steps"],
            "save_final_checkpoint": config["save_final_checkpoint"],
            "log_dir": config["log_dir"],
        }

        run = wandb.init(
            config=wandb_config,
        )
        run_name = run.name

    # Follow the optimizer setup here:
    # https://huggingface.co/EleutherAI/neox-ckpt-pythia-160m-v1/blob/main/160M.yml
    optimizer = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.AdamW,
        lr=config["init_lr"],
    )
    if "pretrained_optimizer_path" in config:
        pretrained_optimizer = torch.load(config["pretrained_optimizer_path"])
        optimizer.load_state_dict(pretrained_optimizer.state_dict())
    print_with_rank(
        rank, count_parameters(pretrained_model), count_optimizer_parameters(optimizer)
    )
    # We use a constant learning rate as the steps we trained on are usually less
    # than 2% of the entire pre-training, which corresponds to very small learning
    # rate change.
    lr_scheduler = get_scheduler(
        "constant", optimizer=optimizer, num_training_steps=num_training_steps
    )

    feature_keys = ["input_ids"]
    epoch = 0
    eval_results = {}
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    mem_df, batch_df = None, None

    print_with_rank(rank, f"Total number of steps per epoch: {num_training_steps}")

    for step, input_batch in enumerate(train_dataloader):
        if (
            rank == 0
            and step % config["eval_every_n_steps"] == 0
            and config["run_eval"]
        ):
            # run evaluation
            model.eval()
            val_metrics = collections.defaultdict(list)
            with torch.no_grad():
                for val_input_batch in val_dataloader:
                    for k in val_input_batch:
                        val_input_batch[k] = val_input_batch[k].to(device)
                    loss, logits, labels = lm_train_step(model, val_input_batch)
                    metrics = compute_metrics(logits, labels)
                    for key in metrics:
                        val_metrics[f"validation-{key}"].append(
                            metrics[key].detach().cpu()
                        )
                    val_metrics["validation-loss"].append(
                        loss.float().detach().cpu().mean()
                    )
            for key in val_metrics:
                val_metrics[key] = float(np.array(val_metrics[key]).mean())
            print_with_rank(
                rank,
                "Epoch %d Step %d: Loss %.4f Accuracy %.4f LR %.2E"
                % (
                    epoch,
                    step,
                    val_metrics["validation-loss"],
                    val_metrics["validation-token_accuracy"],
                    lr_scheduler.get_last_lr()[0],
                ),
            )
            metrics_logger["loss"].append(val_metrics["validation-loss"])
            metrics_logger["accuracy"].append(metrics["token_accuracy"])
            wandb.log(val_metrics, step=step)

        # run training step
        model.train()
        for k in feature_keys:
            input_batch[k] = input_batch[k].to(device)
        loss, logits, labels = lm_train_step(model, input_batch)

        # Perform gradient accumulation if needed.
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # Perform optimizer step
        if (step + 1) % gradient_accumulation_steps == 0:
            if rank == 0 and step % config["log_every_n_steps"] == 0:
                wandb.log(
                    {
                        "step": step,
                        "loss": loss.detach().cpu().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=step,
                )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        del loss, logits, labels
        gc.collect()
        torch.cuda.empty_cache()

        if step == config["stop_after_n_steps"] + 1:
            print_with_rank(rank, f"Early stopping after {step} steps")
            break

        # TODO(mm): debug and change early stopping condition
        # if step == round(
        #     config["inject_every_n"]
        #     * config["total_number_inject"]
        #     / config["training_batch_size"]
        #     + config["inject_every_n"] / 2 / config["training_batch_size"]
        # ):
        #     break

        if step % config["compute_mem_every_n_steps"] == 0:
            # evaluate verbatim memorization length for the single-sequence single-shot experiment
            model.eval()
            sequence = tokenizer.decode(
                tokenizer(config["inject_data"][0]).input_ids[: config["window_size"]]
            )  # there is only one sequence

            # get a dict mapping a sequence to a dict of prefixes and corresponding completions
            sequence_to_memorized = get_memorized_sequences(
                model.module,
                tokenizer,
                [sequence],
                prompt_lengths=None,
                max_output_length=64,
                batch_size=config["eval_batch_size"],
                debug=True,
            )

            # compute maximum memorization length
            # we iterate over all prefixes and check the largest completion
            # we only consider the first sequence as there is only one sequence
            max_mem_length = (
                max(
                    [
                        len(tokenizer(v).input_ids)
                        for k, v in list(sequence_to_memorized.values())[0].items()
                    ]
                )
                if sequence_to_memorized
                else len(sequence_to_memorized)
            )
            print_with_rank(
                rank, f"Step {step} Max verbatim memorized length:", max_mem_length
            )

            ############################################################################################
            # logging
            ############################################################################################
            if rank == 0:
                wandb.log({"max_mem_length": max_mem_length}, step=step)
                # log the completions as a table in wandb
                prefixes_to_completions = list(sequence_to_memorized.values())[0]
                if mem_df is None:
                    # create a new wandb table
                    mem_df = pd.DataFrame(
                        prefixes_to_completions.items(),
                        columns=["prefix", f"completion_{step}"],
                    )
                else:
                    # add a new column to the existing table
                    mem_df[f"completion_{step}"] = prefixes_to_completions.values()

                # log the table
                prefixes_to_completions_table = wandb.Table(
                    dataframe=mem_df
                )  # we have to recreate the table object
                wandb.log({"prefixes_to_completions": prefixes_to_completions_table})
                del prefixes_to_completions_table

                # log the current batch in a table
                batch = input_batch["input_ids"]
                batch_text = tokenizer.batch_decode(batch, skip_special_tokens=True)

                if batch_df is None:
                    # create a new wandb table
                    batch_df = pd.DataFrame(
                        batch_text,
                        columns=[f"batch_at_{step}"],
                    )
                else:
                    # add a new column to the existing table
                    batch_df[f"batch_at_{step}"] = batch_text

                # log the table
                batches_table = wandb.Table(
                    dataframe=batch_df
                )  # we have to recreate the table object
                wandb.log({"batches": batches_table})
                del batches_table
            ############################################################################################

            eval_results[step] = sequence_to_memorized
            del sequence_to_memorized
            gc.collect()
            torch.cuda.empty_cache()
            model.train()  # back to training mode
        
        if rank == 0 and step > 0 and step % config["save_every_n_steps"] == 0:
            output_folder = os.path.join(config["log_dir"], run_name)
            # we have to unwrap the model first
            model.module.save_pretrained(os.path.join(output_folder, f"model_at_step{step}.pt"))

    metrics_logger["verbatim_memorization_length"].append(eval_results)

    return model, metrics_logger, run_name


def run_worker(rank, world_size, config):
    set_seed(0)
    setup_ddp(rank, world_size, config["port"])
    tokenizer = load_model_and_tokenizer(
        config["base_model"], config["hf_cache_dir"], rank, tokenizer_only=True
    )

    print("Constructing training dataset ...")
    train_dataset = NumpyArrayDataset(
        data=config["data"],
        # sample_range=config["training_sample_range"],
        sample_range=None,
        inject_data=config["inject_data"],
        inject_every_n=config["inject_every_n"],
        tokenizer=tokenizer,
        process_id=rank,
    )
    train_dataset.window_size = config["window_size"]

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training_batch_size"],
        shuffle=False,
        sampler=DistributedSampler(train_dataset, rank=rank, shuffle=False),
    )

    # TODO(mm): implement a different way of providing validation data
    # print("Constructing validation dataset ...")
    # val_dataset = NumpyArrayDataset(
    #     data=config["data"], sample_range=config["eval_sample_range"]
    # )
    # val_dataset.window_size = config["window_size"]

    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=config["eval_batch_size"],
    #     shuffle=False,
    #     sampler=DistributedSampler(val_dataset, rank=rank, shuffle=False),
    # )
    val_dataloader = None

    # start distributed training
    model, metrics, run_name = train_distributed_model(
        rank, world_size, train_dataloader, val_dataloader, config
    )

    if rank == 0 and config["save_final_checkpoint"]:
        output_folder = os.path.join(config["log_dir"], run_name)
        # we have to unwrap the model first
        model.module.save_pretrained(os.path.join(output_folder, "model.pt"))

    dist.destroy_process_group()
