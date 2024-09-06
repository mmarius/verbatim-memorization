import argparse
import torch
from torch.utils.data import DataLoader

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


def _compute_batch_perplexity(
    model, input_ids, attention_masks, max_length, stride, device
):
    seq_len = input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        strided_input_ids = input_ids[:, begin_loc:end_loc].to(device)
        strided_attention_masks = attention_masks[:, begin_loc:end_loc].to(device)
        target_ids = strided_input_ids.clone()
        # some of the target ids are padding tokens, let's ignore them when computing the loss
        ignore_ids = target_ids == model.config.pad_token_id
        target_ids[ignore_ids] = -100
        target_ids[:, :-trg_len] = -100

        # print("end_loc", end_loc)
        # print("trg_len", trg_len)
        # print("strided_input_ids.shape", strided_input_ids.shape)
        # print("ignore_ids.shape", ignore_ids.shape)
        # print("target_ids.shape", target_ids.shape)

        with torch.no_grad():
            # TODO(mm): to get the loss per sequence in the batch, move the loss computation outside of the model() call
            outputs = model(
                strided_input_ids, strided_attention_masks, labels=target_ids
            )  # labels are shifted by 1 internally
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())

    return ppl


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

    # iterate over dataset in batches and compute perplexity on each batch
    perplexities = []
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(args.device)  # put data on device
        attention_masks = batch["attention_mask"].to(args.device)  # put data on device
        ppl = _compute_batch_perplexity(
            model,
            input_ids,
            attention_masks,
            max_length=args.max_seq_length,
            stride=args.max_seq_length,  # TODO(mm): for now we just shift by max_seq_length. Change stride to min_context and change implementation accordinglt
            device=args.device,
        )
        perplexities.append(ppl.detach().cpu().item())

    # create a dataframe object to store the perplexities
    df = pd.DataFrame({"perplexity": perplexities})
    df.to_csv(
        f"{args.output_dir}/{args.model_name_or_path.replace('/', '-')}_{args.max_seq_length}_{args.stride}_injection-ppl.csv",
        index=False,
    )
    # save the dataframe object to a csv file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="EleutherAI/pythia-70m"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=256) # Huang et al. (2024) truncate injection sequences after 256 tokens
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    print(args)
    main(args)
