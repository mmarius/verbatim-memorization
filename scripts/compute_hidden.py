import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd


def _load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def _load_injection_data(
    dataset_name="mmosbach/demystifying-verbatim-mem-injection-data",
):
    dataset = load_dataset(dataset_name)["train"]
    return dataset


def main(args):
    # load model and tokenizer
    model, tokenizer = _load_model_and_tokenizer(args.model_name_or_path)
    model.to(args.device)  # put model on device
    if "EleutherAI/pythia" in args.model_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token  # add padding token to the tokenizer
        model.config.pad_token_id = (
            tokenizer.eos_token_id
        )  # add padding token to the model config

    # load injection data
    dataset = _load_injection_data()

    # encode each instance of the dataset
    encoded_dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length,
        ),
        batched=True,
    )

    # create a dataloader from the dataset object
    dataloader = DataLoader(
        encoded_dataset.with_format("torch"), batch_size=args.batch_size, shuffle=False
    )

    # iterate over dataset in batches and get hidden representations
    hidden_reps = []
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(args.device)  # put data on device
        attention_masks = batch["attention_mask"].to(args.device)  # put data on device

        # run forward pass and get hidden representations
        with torch.no_grad():
            outputs = model(input_ids, attention_masks, output_hidden_states=True)
            # print("Number of layers:", len(outputs.hidden_states))
            # print("Hidden sates shape:", outputs.hidden_states[-1].shape)
            # print(outputs.hidden_states[-1].shape)
            hidden_reps.append(
                outputs.hidden_states[-1].cpu().numpy()
            )  # collect hidden representations from the last layer

    # save the hidden representations to disk
    hidden_reps = np.asarray(hidden_reps)
    hidden_reps = hidden_reps.reshape(-1, hidden_reps.shape[-2], hidden_reps.shape[-1])
    # save numpy arrary to disk
    np.save(
        f"{args.output_dir}/{args.model_name_or_path.replace('/', '-')}_{args.max_seq_length}_hidden.npy",
        hidden_reps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="EleutherAI/pythia-160m"
    )
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--max_seq_length", type=int, default=256
    )  # Huang et al. (2024) truncate injection sequences after 256 tokens
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    print(args)
    main(args)
