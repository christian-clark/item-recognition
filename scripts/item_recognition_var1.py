import argparse
import numpy as np
import random
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTNeoXForCausalLM,
    OPTForCausalLM
)

GPT2_MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl"
]

OPT_MODELS = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b"
]

LIST_COUNT = 500
PREFACE = "In the morning, I had a meeting with a group of people including"
CONTINUATION = "In the afternoon, I again encountered"


def generate_name_lists(names_fn, names_per_list):
    names = list()
    for l in open(names_fn):
        names.append(l.strip())
    names = np.array(names)
    name_lists = list()
    for _ in range(LIST_COUNT):
        random_names = np.random.choice(names, names_per_list, replace=False).tolist()
        name_lists.append(random_names)
    return name_lists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names")
    parser.add_argument("model")
    parser.add_argument("--length", '-l', type=int, help="Names per list", default=10)
    parser.add_argument("--checkpoint", '-c', type=int, help="Pythia checkpoint", default=None)
    parser.add_argument("--seed", '-s', type=int, help="random seed")
    parser.add_argument("--gpu", '-g', action="store_true", help="use GPU")
    args = parser.parse_args()
    if args.seed:
        random.seed(args.seed)

    if args.gpu:
        device = "cuda"
    else:
        device = "cpu"

    model_name = args.model
    model_variant = model_name.split('/')[-1]
    # checkpoint should only be given for Pythia models
    checkpoint = args.checkpoint
    if checkpoint:
        assert "pythia" in model_variant
    if model_name in GPT2_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif model_name in OPT_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = OPTForCausalLM.from_pretrained(model_name)
    elif "pythia" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if checkpoint:
            model = GPTNeoXForCausalLM.from_pretrained(
                model_name,
                revision=f"step{checkpoint}"
            )
        else:
            model = GPTNeoXForCausalLM.from_pretrained(model_name)
    else:
        raise Exception("unsupported model variant")

    model.eval()
    model = model.to(device)
    softmax = torch.nn.Softmax(dim=-1)
    bos_id = model.config.bos_token_id

    # pass in add_special_tokens=False for OPT tokenizer
    def tokenize(*args, **kwargs):
        return tokenizer(*args, add_special_tokens=False, **kwargs)

    input_ids = [bos_id] + tokenize(PREFACE).input_ids

    print("names tgtIx baselineSurp surp")
    name_lists  = generate_name_lists(args.names, args.length)

    for name_list in name_lists:
        input_ids = [bos_id] + tokenize(PREFACE).input_ids
        # token indices of each name's first occurrence
        assert len(name_list) >= 3, "logic assumes 3+ names"
        for n_ix, name in enumerate(name_list):
            # no "and" before final conjunct
            if n_ix == len(name_list) - 1:
                name_ids = tokenize(f" {name}").input_ids
                assert len(name_ids) == 1, "multi-token name?"
            else:
                name_ids = tokenize(f" {name},").input_ids
                assert len(name_ids) == 2, "multi-token name?"
            input_ids.extend(name_ids)

        input_ids.extend(tokenize(f" {CONTINUATION}").input_ids)
        if model_name in GPT2_MODELS:
            model_input = torch.tensor(input_ids)
        else:
            model_input = torch.tensor(input_ids).unsqueeze(0)
        model_input = model_input.to(device)
        model_output = model(model_input)
        # dim: length x vocab
        surps = -torch.log2(softmax(model_output.logits.squeeze(0))).to("cpu")

        baseline_input_ids = [bos_id] + tokenize(CONTINUATION).input_ids
        if model_name in GPT2_MODELS:
            model_input = torch.tensor(baseline_input_ids)
        else:
            model_input = torch.tensor(baseline_input_ids).unsqueeze(0)
        model_input = model_input.to(device)
        model_output = model(model_input)
        # dim: length x vocab
        baseline_surps = -torch.log2(softmax(model_output.logits.squeeze(0))).to("cpu")

        for nix, name in enumerate(name_list):
            target_id = tokenize(f" {name}").input_ids
            assert len(target_id) == 1, "multi-token name?"
            target_surp = surps[-1, target_id].item()
            baseline_target_surp = baseline_surps[-1, target_id].item()

            print(
                ','.join(name_list), nix, baseline_target_surp, target_surp
            )



if __name__ == "__main__":
    main()
