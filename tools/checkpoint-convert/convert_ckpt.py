import argparse

import torch
from transformers import AutoModelForCausalLM
from transformers import (
    MixtralForCausalLM,
    MixtralConfig
)
import json

def load_config_from_json(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        config = MixtralConfig.from_dict(config)
    return config

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace transformers model name"
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (`model.pth`)")
    parser.add_argument("--out", type=str, required=True, help="Path to output directory")
    parser.add_argument("--sequence-length", type=int, required=True)
    args = parser.parse_args()

    if "Mixtral_pretrain" in args.model:
        config = load_config_from_json(config_file = "/root/moe-recipes/src/llama_recipes/config.json") 
        config.attn_implementation = "flash_attention_2"
        config.max_position_embeddings = 1024
        config.torch_dtype=torch.bfloat16
        config.use_cache=False
        model = MixtralForCausalLM(config)
    else:
        print(f"Loading HF model: {args.model}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            max_position_embeddings=args.sequence_length,
        )

    print(f"Loading CKPT: {args.ckpt}", flush=True)
    state_dict = torch.load(args.ckpt, map_location="cpu")

    print("Loading state dict into HF model", flush=True)
    model.load_state_dict(state_dict)

    print("Saving HF model", flush=True)
    model.save_pretrained(args.out, safe_serialization=True)


if __name__ == "__main__":
    main()
