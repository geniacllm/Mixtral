from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralForCausalLM,
    MistralConfig,
    MixtralForCausalLM,
    MixtralConfig,
    AutoModelForCausalLM,
    AutoConfig
)
from llama_recipes.utils.distributed import get_rank, is_rank_0
import torch
import json
from megatron_lm.megatron.global_vars import get_args
from transformers.integrations import is_deepspeed_zero3_enabled
import os
home_directory = os.getenv('HOME')
recipe_dir = os.getenv('ucllm_nedo_dev')

def load_config_from_json(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        config = MixtralConfig.from_dict(config)
    return config

def get_model(
    model_name: str, use_cache: bool = False
) -> LlamaForCausalLM | MistralForCausalLM | MixtralForCausalLM | AutoModelForCausalLM:
    """return CausalLM model

    Args:
        model_name: str
        use_cache (bool, optional):

    Raises:
        NotImplementedError: currently only supports LlamaForCausalLM and MistralForCausalLM

    Returns:
        LlamaForCausalLM | MistralForCausalLM: PyTorch model
    """
    args = get_args()

    if "Llama" in model_name:
        if args.low_cpu_fsdp:
            """
            for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
            this avoids cpu oom when loading large models like llama 70B, in which case
            model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some communications
            overhead.
            """
            if is_rank_0():
                model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True if args.quantization else None,
                    device_map="auto" if args.quantization else None,
                    use_cache=use_cache,
                )
            else:
                llama_config = LlamaConfig.from_pretrained(model_name)
                llama_config.use_cache = use_cache
                with torch.device("meta"):
                    model = LlamaForCausalLM(llama_config)

        else:
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True if args.quantization else None,
                device_map="auto" if args.quantization else None,
                use_cache=use_cache,
            )

        return model  # type: ignore

    elif "Mistral" in model_name or "mistral" in model_name:
        # If using torch.device("meta"), FSDP training hang
        # FYI: https://github.com/iwiwi/epochraft-hf-fsdp/pull/10#issuecomment-1803360147
        # https://github.com/pytorch/pytorch/issues/105840 are maybe helpful
        mistral_max_length: int = args.seq_length
        sliding_window: int = args.sliding_window_size
        assert sliding_window == 4096

        model = MistralForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True if args.quantization else None,
            device_map="auto" if args.quantization else None,
            use_cache=use_cache,
            sliding_window=sliding_window,
            max_position_embeddings=mistral_max_length,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        )

        return model  # type: ignore

    elif "Mixtral_pretrain" in model_name:
        
        config = load_config_from_json(config_file = recipe_dir+"/src/llama_recipes/config.json") 
        config.attn_implementation = "flash_attention_2"
        config.max_position_embeddings = args.seq_length
        config.torch_dtype=torch.float16
        config.use_cache=use_cache
        
        model = MixtralForCausalLM(config)
        
        return model  # type: ignore
    
    elif "Mixtral" in model_name:
        model = MixtralForCausalLM.from_pretrained(
            model_name,
            # device_map="auto", (これがあるとダメ)
            attn_implementation="flash_attention_2",
            max_position_embeddings=args.seq_length,
            # ref: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/config.json#L19
            output_router_logits=args.output_router_logits,
            torch_dtype=torch.float16,
            use_cache=use_cache,
        )

        return model  # type: ignore

    elif "calm2-7b" in model_name:
        # calm2-7b is compatible with LlamaForCausalLM
        # https://huggingface.co/cyberagent/calm2-7b/blob/main/config.json
        if args.low_cpu_fsdp:
            """
            for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
            this avoids cpu oom when loading large models like llama 70B, in which case
            model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some communications
            overhead.
            """
            if is_rank_0():
                model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True if args.quantization else None,
                    device_map="auto" if args.quantization else None,
                    use_cache=use_cache,
                )
            else:
                llama_config = LlamaConfig.from_pretrained(model_name)
                llama_config.use_cache = use_cache
                with torch.device("meta"):
                    model = LlamaForCausalLM(llama_config)

        else:
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True if args.quantization else None,
                device_map="auto" if args.quantization else None,
                use_cache=use_cache,
            )

        return model  # type: ignore

    elif "japanese-stablelm-base-alpha-7b" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto" if args.quantization else None,
            use_cache=use_cache,
        )

        return model  # type: ignore

    elif "stockmark-13b" in model_name:
        # stockmark-13b is compatible with LlamaForCausalLM
        # https://huggingface.co/stockmark/stockmark-13b/blob/main/config.json
        if args.low_cpu_fsdp:
            """
            for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
            this avoids cpu oom when loading large models like llama 70B, in which case
            model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some communications
            overhead.
            """
            if is_rank_0():
                model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True if args.quantization else None,
                    device_map="auto" if args.quantization else None,
                    use_cache=use_cache,
                )
            else:
                llama_config = LlamaConfig.from_pretrained(model_name)
                llama_config.use_cache = use_cache
                with torch.device("meta"):
                    model = LlamaForCausalLM(llama_config)

        else:
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True if args.quantization else None,
                device_map="auto" if args.quantization else None,
                use_cache=use_cache,
            )

        return model  # type: ignore

    elif "plamo-13b" in model_name:
        if args.low_cpu_fsdp:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True if args.quantization else None,
                device_map="auto" if args.quantization else None,
                use_cache=use_cache,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True if args.quantization else None,
                device_map="auto" if args.quantization else None,
                use_cache=use_cache,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

        return model  # type: ignore

    elif "llm-jp-13b-v1.0" in model_name:
        # llm-jp 13b v1.0 is compatible with GPT2
        # https://huggingface.co/llm-jp/llm-jp-13b-v1.0/blob/main/config.json
        if args.low_cpu_fsdp:
            if get_rank() == 0:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True if args.quantization else None,
                    device_map="auto" if args.quantization else None,
                    use_cache=use_cache,
                    torch_dtype=torch.float16,
                )
            else:
                with torch.device("meta"):
                    model = AutoModelForCausalLM.from_config(
                        AutoConfig.from_pretrained(
                            model_name,
                            device_map="auto" if args.quantization else None,
                            use_cache=use_cache,
                            torch_dtype=torch.float16,
                        ),
                    )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True if args.quantization else None,
                device_map="auto" if args.quantization else None,
                use_cache=use_cache,
                torch_dtype=torch.float16,
            )

        return model  # type: ignore

    elif "ELYZA-japanese-Llama-2-7b" in model_name:
        # ELYZA-japanese-Llama-2-7b is compatible with LlamaForCausalLM
        # https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b/blob/main/config.json
        if args.low_cpu_fsdp:
            """
            for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
            this avoids cpu oom when loading large models like llama 70B, in which case
            model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some communications
            overhead.
            """
            if is_rank_0():
                model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True if args.quantization else None,
                    device_map="auto" if args.quantization else None,
                    use_cache=use_cache,
                )
            else:
                llama_config = LlamaConfig.from_pretrained(model_name)
                llama_config.use_cache = use_cache
                with torch.device("meta"):
                    model = LlamaForCausalLM(llama_config)

        else:
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True if args.quantization else None,
                device_map="auto" if args.quantization else None,
                use_cache=use_cache,
            )

        return model  # type: ignore

    elif "japanese-stablelm-base-ja_vocab-beta-7b" in model_name:
        # japanese-stablelm-base-ja_vocab-beta-7b is compatible with LlamaForCausalLM
        # https://huggingface.co/stabilityai/japanese-stablelm-base-ja_vocab-beta-7b/blob/main/config.json
        if args.low_cpu_fsdp:
            """
            for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
            this avoids cpu oom when loading large models like llama 70B, in which case
            model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some communications
            overhead.
            """
            if is_rank_0():
                model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True if args.quantization else None,
                    device_map="auto" if args.quantization else None,
                    use_cache=use_cache,
                )
            else:
                llama_config = LlamaConfig.from_pretrained(model_name)
                llama_config.use_cache = use_cache
                with torch.device("meta"):
                    model = LlamaForCausalLM(llama_config)

        else:
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True if args.quantization else None,
                device_map="auto" if args.quantization else None,
                use_cache=use_cache,
            )

        return model  # type: ignore

    elif "japanese-stablelm-base-beta" in model_name:
        # stabilityai/japanese-stablelm-base-beta is compatible with LlamaForCausalLM
        # https://huggingface.co/stabilityai/japanese-stablelm-base-beta-7b/blob/main/config.json
        if args.low_cpu_fsdp:
            """
            for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
            this avoids cpu oom when loading large models like llama 70B, in which case
            model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some communications
            overhead.
            """
            if is_rank_0():
                model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True if args.quantization else None,
                    device_map="auto" if args.quantization else None,
                    use_cache=use_cache,
                )
            else:
                llama_config = LlamaConfig.from_pretrained(model_name)
                llama_config.use_cache = use_cache
                with torch.device("meta"):
                    model = LlamaForCausalLM(llama_config)

        else:
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True if args.quantization else None,
                device_map="auto" if args.quantization else None,
                use_cache=use_cache,
            )

        return model  # type: ignore

    else:
        raise NotImplementedError("model not implemented")
