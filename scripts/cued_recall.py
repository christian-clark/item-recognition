import argparse
import numpy as np
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
INITIAL = "Please try to remember these sentences."
TEMPLATE = "{} lives in {}."
INTERVENING = "Now, after you received all this information, try to concentrate, drink a cup of coffee, go for a walk. Then please complete the following sentence, based on what you read earlier."
FINAL = "{} lives in"


def generate_pairs(names_fn, cities_fn, pairs_per_list):
    names = list()
    for l in open(names_fn):
        names.append(l.strip())

    cities = list()
    for l in open(cities_fn):
        cities.append(l.strip())

    names = np.array(names)
    cities = np.array(cities)
    pair_lists = list()
    for _ in range(LIST_COUNT):
        random_names = np.random.choice(names, pairs_per_list, replace=False).tolist()
        random_cities = np.random.choice(cities, pairs_per_list, replace=False).tolist()
        pair_lists.append(list(zip(random_names, random_cities)))
    return pair_lists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names")
    parser.add_argument("cities")
    parser.add_argument("model")
    parser.add_argument("--length", '-l', type=int, help="Number of pairs per list", default=10)
    parser.add_argument("--checkpoint", '-c', type=int, help="Pythia checkpoint", default=None)
    parser.add_argument("--seed", '-s', type=int, help="random seed")
    parser.add_argument("--gpu", '-g', action="store_true", help="use GPU")
    args = parser.parse_args()
    if args.seed:
        np.random.seed(args.seed)
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
                cache_dir=f"../cached_lms/{model_variant}/step{checkpoint}",
            )
        else:
            model = GPTNeoXForCausalLM.from_pretrained(model_name)
    else:
        raise Exception("unsupported model variant")

    model.eval()
    model = model.to(device)
    softmax = torch.nn.Softmax(dim=-1)
    if "mamba2" in model_variant or "mpt" in args.model:
        bos_id = tokenizer.bos_token_id
    else:
        bos_id = model.config.bos_token_id

    # pass in add_special_tokens=False for OPT tokenizer
    def tokenize(*args, **kwargs):
        return tokenizer(*args, add_special_tokens=False, **kwargs)

    print("names cities pairIx surp1 surp2")
    pair_lists = generate_pairs(args.names, args.cities, args.length)
    for pair_list in pair_lists:
        names = [p[0] for p in pair_list]
        cities = [p[1] for p in pair_list]
        input_ids = [bos_id] + tokenize(INITIAL).input_ids
        # token indices of the first instance of each city
        city_ixs = list()
        for noun, city in pair_list:
            sent = TEMPLATE.format(noun, city)
            sent_ids = tokenize(f" {sent}").input_ids
            assert len(sent_ids) == 5 # X lives in Y.
            input_ids.extend(sent_ids)
            # this is the location of the token right before the city, i.e. "in".
            # the output layer at this index will have surprisal for the city
            city_ixs.append(len(input_ids) - 3)
        input_ids.extend(tokenize(f" {INTERVENING}").input_ids)

        for pix, (noun, city) in enumerate(pair_list):
            curr_input_ids = input_ids[:]
            final_sent = FINAL.format(noun)
            curr_input_ids.extend(tokenize(f" {final_sent}").input_ids)
            if model_name in GPT2_MODELS:
                model_input = torch.tensor(curr_input_ids)
            else:
                model_input = torch.tensor(curr_input_ids).unsqueeze(0)
            model_input = model_input.to(device)
            #print("input:", tokenizer.convert_ids_to_tokens(input_ids), file=sys.stderr)
            model_output = model(model_input)
            # dim: length x vocab
            surps = -torch.log2(softmax(model_output.logits.squeeze(0))).to("cpu")
            target_id = tokenize(f" {city}").input_ids
            target_ix = city_ixs[pix]
            target_surp1 = surps[target_ix, target_id].item()
            target_surp2 = surps[-1, target_id].item()
            assert len(target_id) == 1, "multi-token city?"
            print(','.join(names), ','.join(cities), pix, target_surp1, target_surp2)


if __name__ == "__main__":
    main()
