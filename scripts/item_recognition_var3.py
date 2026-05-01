# like the list-first prompt except no "and" before the final list item
import argparse
import numpy as np
import random
import sys
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
#LIST_COUNT = 5
PREFACE = "The cities I have traveled to include"
INTERVENING = "Next year, I will again be visiting"
BARE_PROMPT = "Next year, I will again be visiting"


def generate_city_lists(cities_fn, cities_per_list):
    cities = list()
    for l in open(cities_fn):
        cities.append(l.strip())
    cities = np.array(cities)
    city_lists = list()
    for _ in range(LIST_COUNT):
        random_cities = np.random.choice(cities, cities_per_list, replace=False).tolist()
        city_lists.append(random_cities)
    return city_lists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cities")
    parser.add_argument("model")
    parser.add_argument("--length", '-l', type=int, help="cities per list", default=10)
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
    elif "linear" in model_variant:
        from transformers import GPTNeoXLinearForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision="word")
        model = GPTNeoXLinearForCausalLM.from_pretrained(model_name, revision="word")
    elif "mamba2" in model_variant:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        assert args.gpu, "GPU required for Mamba2"
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = MambaLMHeadModel.from_pretrained(model_name)
    elif "alibi" in model_variant:
        from transformers import GPTNeoXAlibiForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision="word")
        model = GPTNeoXAlibiForCausalLM.from_pretrained(model_name, revision="word")
    elif "fleeting" in model_variant:
        from transformers import GPTNeoXFleetingForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision="word")
        model = GPTNeoXFleetingForCausalLM.from_pretrained(model_name, revision="word")
    elif "vanilla" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision="word")
        model = GPTNeoXForCausalLM.from_pretrained(model_name, revision="word")
    elif "pythia" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if checkpoint:
            model = GPTNeoXForCausalLM.from_pretrained(
                model_name,
                revision=f"step{checkpoint}",
                cache_dir=f"/users/PAS2157/ceclark/git/cued-recall/cached_lms/{model_variant}/step{checkpoint}",
            )
        else:
            model = GPTNeoXForCausalLM.from_pretrained(model_name)
    else:
        raise Exception("unsupported model variant")

    model.eval()
    model = model.to(device)
    softmax = torch.nn.Softmax(dim=-1)
    if "mamba2" in args.model:
        bos_id = tokenizer.bos_token_id
    else:
        bos_id = model.config.bos_token_id

    # pass in add_special_tokens=False for OPT tokenizer
    def tokenize(*args, **kwargs):
        return tokenizer(*args, add_special_tokens=False, **kwargs)

    print("cities tgtIx baselineSurp surp surpRatio")
    city_lists  = generate_city_lists(args.cities, args.length)

    for city_list in city_lists:
        input_ids = [bos_id] + tokenize(PREFACE).input_ids
        # token indices of each city's first occurrence
        #city_locs = list()
        assert len(city_list) >= 3, "logic assumes 3+ cities"
        for n_ix, city in enumerate(city_list):
            # no "and" before final conjunct
            if n_ix == len(city_list) - 1:
                city_ids = tokenize(f" {city}").input_ids
                assert len(city_ids) == 1, "multi-token city?"
            else:
                city_ids = tokenize(f" {city},").input_ids
                assert len(city_ids) == 2, "multi-token city?"
            input_ids.extend(city_ids)

        # add period after last city
        input_ids.extend(tokenize(f". {INTERVENING}").input_ids)
        print("input:", tokenizer.convert_ids_to_tokens(input_ids), file=sys.stderr)
        if model_name in GPT2_MODELS:
            model_input = torch.tensor(input_ids)
        else:
            model_input = torch.tensor(input_ids).unsqueeze(0)
        model_input = model_input.to(device)
        model_output = model(model_input)
        # dim: length x vocab
        surps = -torch.log2(softmax(model_output.logits.squeeze(0))).to("cpu")

        baseline_input_ids = [bos_id] + tokenize(BARE_PROMPT).input_ids
        if model_name in GPT2_MODELS:
            model_input = torch.tensor(baseline_input_ids)
        else:
            model_input = torch.tensor(baseline_input_ids).unsqueeze(0)
        model_input = model_input.to(device)
        model_output = model(model_input)
        # dim: length x vocab
        baseline_surps = -torch.log2(softmax(model_output.logits.squeeze(0))).to("cpu")

        for nix, city in enumerate(city_list):
            target_id = tokenize(f" {city}").input_ids
            assert len(target_id) == 1, "multi-token city?"
            target_surp = surps[-1, target_id].item()
            baseline_target_surp = baseline_surps[-1, target_id].item()

            print(
                ','.join(city_list), nix, baseline_target_surp, target_surp,
                target_surp/baseline_target_surp
            )


if __name__ == "__main__":
    main()
