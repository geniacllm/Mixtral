import os
import time
import wandb
from pkg_resources import packaging  # type: ignore
from contextlib import nullcontext

import torch
import torch.cuda.nccl as nccl
from torch import distributed as torch_distributed  # noqa: F401
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn.utils import clip_grad_norm_  # type: ignore

from llama_recipes.policies import fpSixteen, bfSixteen, bfSixteen_mixed, get_decoder_layer_wrapper
from llama_recipes.utils.wandb_utils import log_model_info, log_wandb
from llama_recipes.utils.checkpoint import save_checkpoint, get_latest_iteration

from typing import Optional, Any
from accelerate import Accelerator
from megatron_lm.megatron.global_vars import get_args, get_tokenizer

from transformers.modeling_outputs import MoeCausalLMOutputWithPast


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def parse_skip_batch(args: list[int]) -> list[tuple[int, int]]:
    if len(args) % 2 != 0:
        raise ValueError("The '--skip-batch' option requires an even number of arguments.")

    return [(int(args[i]), int(args[i + 1])) for i in range(0, len(args), 2)]


def train(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer: torch.optim.AdamW,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    gradient_accumulation_steps: int,
    accelerator: Accelerator,
    local_rank: Optional[int] = None,
    rank: Optional[int] = None,
) -> None:
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps:
            The number of steps to accumulate gradients before performing
            a backward/update operation
        local_rank: The rank of the current node in a distributed setting
        rank:

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    args = get_args()

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = local_rank if local_rank is not None else 0
    autocast = torch.cuda.amp.autocast if args.fp16 else nullcontext  # type: ignore

    # set model info
    if rank == 0 and args.wandb_name:
        log_model_info(model.module)

    iteration: int = args.iteration
    real_batch_size: int = args.micro_batch_size
    real_seq_len: int = args.seq_length

    # cyclic iter
    train_dataloader = iter(cyclic_iter(train_dataloader)) if args.dataset_cyclic else iter(train_dataloader)
    eval_dataloader = iter(cyclic_iter(eval_dataloader)) if args.dataset_cyclic else iter(eval_dataloader)
    if args.dataset_cyclic:
        print("CAUTION: dataloader cyclic feature enabled.")

    # skip batch
    skip_batches: list[tuple[int, int]] = []
    consumed_iters: int = iteration
    if args.skip_batch:
        skip_batches = parse_skip_batch(args.skip_batch)

    while iteration < args.train_iters:
        # skip batch logic
        while len(skip_batches) > 0 and skip_batches[0][0] <= consumed_iters <= skip_batches[0][1]:
            next(train_dataloader)
            consumed_iters += 1
        if len(skip_batches) > 0 and skip_batches[0][1] < consumed_iters:
            skip_batches.pop(0)

        iteration_start_time = time.perf_counter()

        model.train()
        total_loss: float = 0.0
        total_load_balancing_loss: float = 0.0

        for _ in range(gradient_accumulation_steps):

            batch = next(train_dataloader)

            #応急処置
            max_value = torch.max(batch["input_ids"])
            while max_value>32000:
              batch = next(train_dataloader)
              max_value = torch.max(batch["input_ids"])
              #print(max_value)

            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
          
            with autocast():
                output: MoeCausalLMOutputWithPast = model(**batch)
                loss: torch.FloatTensor = output.loss  # type: ignore
                load_balancing_loss: torch.FloatTensor = output.aux_loss  # type: ignore
            loss = loss / gradient_accumulation_steps  # type: ignore
            load_balancing_loss = load_balancing_loss / gradient_accumulation_steps  # type: ignore

            # ref: https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py#L167
            # doesn't work if use loss.backward()
            accelerator.backward(loss)

            total_loss += loss.item()
            total_load_balancing_loss += load_balancing_loss.item()

        # gradient clipping
        if args.grad_clip_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        real_batch_size: int = batch["input_ids"].shape[0]  # type: ignore
        real_seq_len: int = batch["input_ids"].shape[1]  # type: ignore

        # gradient accumulation end
        iteration += 1
        consumed_iters += 1

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if args.wandb_name:
            avg_loss: torch.Tensor = torch.tensor(total_loss).to(local_rank)  # type: ignore
            torch_distributed.all_reduce(tensor=avg_loss, op=torch_distributed.ReduceOp.SUM)
            avg_loss = avg_loss / world_size

            if rank == 0:
                log_wandb(
                    real_batch_size=real_batch_size,
                    real_seq_len=real_seq_len,
                    model=model.module,
                    accumulation_loss=avg_loss,  # type: ignore
                    load_balancing_loss=total_load_balancing_loss,
                    optimizer=optimizer.optimizer,  # type: ignore
                    iteration=iteration,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    world_size=world_size,
                    iteration_start_time=iteration_start_time,
                )
            total_loss = 0.0
            total_load_balancing_loss = 0.0
            iteration_start_time = time.perf_counter()

        if (iteration) % args.save_interval == 0:
            # checkpoint save
            save_checkpoint(
                model=model,  # type: ignore
                optimizer=optimizer,
                scheduler=lr_scheduler,
                path=args.save,
                iteration=iteration,
            )

    torch_distributed.barrier()
    save_checkpoint(
        model=model,  # type: ignore
        optimizer=optimizer,
        scheduler=lr_scheduler,
        path=args.save,
        iteration=iteration,
    )
    torch_distributed.barrier()


def evaluation(
    model,
    eval_dataloader,
    local_rank: int,
    wandb_log: bool = False,
):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting

    Returns: eval_ppl, eval_epoch_loss
    """
    world_size = int(os.environ["WORLD_SIZE"])
    args = get_args()

    model.eval()
    eval_loss = 0.0
    iteration = 0

    while iteration < args.eval_iters:
        iteration += 1
        batch = next(eval_dataloader)

        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)

        with torch.no_grad():
            # Forward pass and compute loss
            outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()

    torch_distributed.all_reduce(eval_loss, op=torch_distributed.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss: float = eval_loss / args.eval_iters / world_size
    eval_ppl = torch.exp(eval_epoch_loss)  # type: ignore

    # Print evaluation metrics
    if torch_distributed.get_rank() == 0:
        print(f" eval ppl={eval_ppl}, eval loss={eval_epoch_loss}")

    return eval_ppl, eval_epoch_loss


def parse_layer_ranges(layer_ranges: str) -> list:
    """parse layer ranges

    Args:
        layer_ranges (str): (ex: '1-5,7-9')

    Returns:
        list: freezed layers' list
    """
    layers_to_freeze: list[int] = []
    for part in layer_ranges.split(','):
        if '-' in part:
            start, end = part.split('-')
            layers_to_freeze.extend(range(int(start), int(end) + 1))
        else:
            layers_to_freeze.append(int(part))
    return layers_to_freeze


def freeze_transformer_layers(model, layer_ranges: str) -> None:
    """freeze transformer layers

    Args:
        model:
        layer_ranges: str
    """
    layers_to_freeze = parse_layer_ranges(layer_ranges)
    for i, layer in enumerate(model.model.layers):
        if i in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False


def check_frozen_layers_peft_model(model) -> None:
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup_environ_flags(rank: int) -> None:
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if rank == 0:
        print("--> Running with torch torch_distributed debug set to detail")


def cleanup() -> None:
    """Clean up the process group after training"""
    torch_distributed.destroy_process_group()


def clear_gpu_cache(rank: Optional[int] = None) -> None:
    """Clear the GPU cache for all ranks"""
    if torch_distributed.get_rank() == 0:
        print("Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model) -> dict[Any, Any]:
    """Get the data types of model parameters"""
    parameter_dtypes: dict[Any, Any] = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, model_name: str, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        rank (int, optional): Current process's rank. Defaults to 0.
    """

    if rank == 0:
        print(f"--> Model {model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {model_name} has {total_params / 1e6} Million params\n")


def get_policies(rank: int, model_name: str):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support: bool = (
        torch.version.cuda  # type: ignore
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)  # type: ignore
        and torch_distributed.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    mixed_precision_policy = None
    wrapping_policy = None

    args = get_args()

    # Mixed precision
    if args.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not args.fp16 and args.param_dtype == "fp32":
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print("\nBFloat16 enabled for mixed precision - using bfSixteen_mixed policy\n", flush=True)
        elif bf16_ready and not args.fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print("\nBFloat16 enabled for mixed precision - using bfSixteen policy\n", flush=True)
        elif args.fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print("\nFP16 enabled\n", flush=True)
        else:
            print("bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_decoder_layer_wrapper(model_name=model_name)
    return mixed_precision_policy, wrapping_policy
