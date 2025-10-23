import argparse
import datetime
import json
import os
from functools import partial

import lightning as L
import shortuuid
import torch
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
from lightning.fabric.plugins.precision import FSDPPrecision
from lightning.fabric.strategies import ModelParallelStrategy
from lightning.fabric.utilities.data import AttributeDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from data.datamodule import TextDataModule
from engine.gpt_training import print_rank_zero, train
from engine.lr_schedulers import init_lr_scheduler
from models import GPTNeoXLayer, LlamaDecoderLayer, Olmo2DecoderLayer
from models.language_model import init_model
from models.tp_utils import parallelize_model
from optimizers import init_optimizer
from strategies.fsdp import FSDPStrategy


def main(
    model_config,
    optim_config,
    dataset,
    run_name,
    logdir,
    saved_ckpt_path,
    ckpt_path=None,
    devices="auto",
    num_nodes=1,
    global_bs=1024,
    training_max_seq_len=2048,
    train_bs=16,
    val_bs=None,
    strategy="ddp",
    dp_size=0,
    tp_size=0,
    grad_clip_val=1,
    precision="bf16-mixed",
    ckpt_every=1000,
    val_check_every=250,
    max_val_batches=10,
    log_every_n_steps=25,
    num_val_samples=1000,
    use_profiler=False,
    detect_anomaly=False,
    load_mode="continue",
    act_ckpt=False,
    init_head=False,
    date_in_log=False,
    seed=42,
) -> None:
    main_fn_args = {**locals()}

    val_bs = train_bs if val_bs is None else val_bs

    date_prefix = ""
    if date_in_log:
        dt = datetime.datetime.now()
        date_prefix = dt.strftime("%y/%m/%d") + "/"

    version_name = f"{run_name}_{os.environ.get('SLURM_JOB_ID', shortuuid.uuid()[:8])}"
    loggers = TensorBoardLogger(root_dir=logdir, version=f"{date_prefix}{version_name}")

    plugins = []
    if "SLURM_NTASKS" in os.environ:
        plugins.append(SLURMEnvironment())

    strategy_arg = strategy
    if "fsdp" in strategy:
        if strategy == "fsdp_grad_op":
            sharding_strategy = "SHARD_GRAD_OP"
        elif strategy == "fsdp_hybrid":
            sharding_strategy = "HYBRID_SHARD"
        else:
            sharding_strategy = "FULL_SHARD"

        layers = {Olmo2DecoderLayer, LlamaDecoderLayer, GPTNeoXLayer}

        strategy = FSDPStrategy(
            auto_wrap_policy=layers,
            sharding_strategy=sharding_strategy,
            activation_checkpointing_policy=layers if act_ckpt else None,
            precision=FSDPPrecision(precision),
            # process_group_backend="gloo"
        )
    elif strategy == "tp":
        strategy = ModelParallelStrategy(
            parallelize_fn=parallelize_model,
            data_parallel_size=dp_size if dp_size else "auto",
            tensor_parallel_size=tp_size if tp_size else "auto",
        )

    fabric = L.Fabric(
        num_nodes=num_nodes,
        precision=precision,
        strategy=strategy,
        loggers=loggers,
        plugins=plugins,
    )
    fabric.launch()
    fabric.seed_everything(seed)

    if strategy_arg == "tp":
        model_parallel_ratio = 1.0 / strategy.num_processes
    else:
        model_parallel_ratio = 1.0

    fabric_print_rank_zero = partial(print_rank_zero, fabric=fabric)

    ckpt_dir = f"{saved_ckpt_path}/{version_name}"

    ckpt_dir = fabric.broadcast(ckpt_dir)

    if fabric.is_global_zero:
        os.makedirs(ckpt_dir, exist_ok=True)

    fabric.barrier()

    if int(os.getenv("SLURM_RESTART_COUNT", 0)) > 0 and "requeue.ckpt" in os.listdir(
        ckpt_dir
    ):
        # TODO: support latest ckpt restart
        ckpt_path = f"{ckpt_dir}/requeue.ckpt"
        load_mode = "continue"
        fabric_print_rank_zero(f"Requeue checkpoint will be loaded from {ckpt_path}.")

    datamodule = TextDataModule(
        dataset,
        max_seq_len=training_max_seq_len,
        num_val_samples=num_val_samples,
        train_batch_size=train_bs,
        val_batch_size=val_bs,
        num_proc=1,
    )
    datamodule.setup(stage="fit")
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        datamodule.train_dataloader(), datamodule.val_dataloader()
    )

    model_config = json.load(open(model_config, "rb"))
    optim_config = json.load(open(optim_config, "rb"))

    if fabric.is_global_zero:
        tb_experiment = fabric.loggers[0].experiment
        tb_experiment.add_text("model_config", json.dumps(model_config))
        tb_experiment.add_text("optim_config", json.dumps(optim_config))
        tb_experiment.add_text("training_config", json.dumps(main_fn_args))
        tb_experiment.add_text(
            "slurm_config",
            json.dumps({k: v for k, v in os.environ.items() if "SLURM" in k}),
        )

    empty_init = strategy != "ddp" and not init_head
    if empty_init:
        fabric_print_rank_zero(
            "Using meta initialization for memory-efficiency.", flush=True
        )

    with fabric.init_module(empty_init=empty_init):
        model = init_model(**model_config)

    model = fabric.setup(model)
    model.init_weights()
    model.train()

    fabric_print_rank_zero("Model initialized and setup.", flush=True)

    optimizer = init_optimizer(list(model.named_parameters()), **optim_config)
    lr_scheduler = init_lr_scheduler(optimizer, **optim_config)

    optimizer = fabric.setup_optimizers(optimizer)

    gpus_by_node = torch.cuda.device_count()

    if (global_bs % (model_parallel_ratio * gpus_by_node * num_nodes * train_bs)) != 0:
        raise ValueError(
            f"Requested a global batch size of {global_bs} on {num_nodes}x{train_bs}x{gpus_by_node}x{model_parallel_ratio} (nodes x train_bs x gpus_per_node x model_parallel_ratio) GPUs but it's not a multiple. Minimum batch size should be {gpus_by_node * num_nodes * train_bs}."
        )
    elif (model_parallel_ratio * gpus_by_node * num_nodes * train_bs) > global_bs:
        raise ValueError(
            f"Global batch size {global_bs} is too small for {num_nodes}x{train_bs}x{gpus_by_node}x{model_parallel_ratio} (nodes x train_bs x gpus_per_node x model_parallel_ratio) GPUs."
        )
    accu_grad_batches = int(
        global_bs // (model_parallel_ratio * gpus_by_node * num_nodes * train_bs)
    )

    fabric_print_rank_zero(
        f"Train BS: {train_bs} | Grad. accumulating factor: {accu_grad_batches} | Val BS: {val_bs}"
    )

    state = AttributeDict(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        step_count=0,
    )

    total_nb_steps = optim_config["total_nb_steps"]

    if ckpt_path is not None:
        fabric_print_rank_zero(f"Loading checkpoint from {ckpt_path}...", flush=True)

        if load_mode == "model_only":
            partial_state = AttributeDict(model=model)
            fabric.load(ckpt_path, partial_state, strict=False)
            state.model = partial_state.model

        elif load_mode == "model_data":
            partial_state = AttributeDict(
                model=model, train_dataloader=train_dataloader
            )
            fabric.load(ckpt_path, partial_state, strict=False)
            state.model = partial_state.model
            state.train_dataloader = partial_state.train_dataloader

        elif load_mode == "model_opt_data":
            partial_state = AttributeDict(
                model=model, train_dataloader=train_dataloader, optimizer=optimizer
            )
            fabric.load(ckpt_path, partial_state, strict=False)
            state.model = partial_state.model
            state.optimizer = partial_state.optimizer
            state.train_dataloader = partial_state.train_dataloader

        elif load_mode == "cooldown":
            if init_head:
                partial_state = AttributeDict(
                    model=model, train_dataloader=train_dataloader, step_count=0
                )
            else:
                partial_state = AttributeDict(
                    model=model,
                    train_dataloader=train_dataloader,
                    optimizer=optimizer,
                    step_count=0,
                )
            fabric.load(ckpt_path, partial_state, strict=False)
            state.model = partial_state.model
            state.train_dataloader = partial_state.train_dataloader
            if not init_head:
                state.optimizer = partial_state.optimizer
            state.step_count = partial_state.step_count
            total_nb_steps += state.step_count

        elif load_mode == "transfer":
            partial_state = AttributeDict(model=model, optimizer=optimizer)
            fabric.load(ckpt_path, partial_state, strict=False)
            state.model = partial_state.model
            state.optimizer = partial_state.optimizer

        elif load_mode == "transfer_with_lr":
            partial_state = AttributeDict(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                step_count=0,
            )
            fabric.load(ckpt_path, partial_state, strict=False)
            fabric.barrier()

            if hasattr(state, "lr_scheduler"):
                if isinstance(partial_state.lr_scheduler, dict):
                    fabric_print_rank_zero(
                        "Loading lr_scheduler state dict from checkpoint."
                    )
                    lr_scheduler.load_state_dict(partial_state.lr_scheduler)
                else:
                    fabric_print_rank_zero("Loading lr_scheduler from checkpoint.")
                    lr_scheduler = partial_state.lr_scheduler

            state.model = partial_state.model
            state.optimizer = partial_state.optimizer
            state.step_count = partial_state.step_count

        else:
            if load_mode != "continue":
                print(
                    f"Unreckognized load mode {load_mode}. Loading checkpoint in `continue` mode by default. All states will be loaded."
                )
            fabric.load(ckpt_path, state, strict=False)
            fabric.barrier()

            if hasattr(state, "lr_scheduler"):
                if isinstance(state.lr_scheduler, dict):
                    fabric_print_rank_zero(
                        "Loading lr_scheduler state dict from checkpoint."
                    )
                    lr_scheduler.load_state_dict(state.lr_scheduler)
                else:
                    fabric_print_rank_zero("Loading lr_scheduler from checkpoint.")
                    lr_scheduler = state.lr_scheduler

        if init_head:

            with FSDP.summon_full_params(state.model, writeback=True):
                embs = state.model.lm_model.get_input_embeddings()
                state.model.lm_model.lm_head.load_state_dict(embs.state_dict())

            fabric.barrier()

        if state.step_count > 0:
            fabric_print_rank_zero(
                f"Restarting training from step {state.step_count}", flush=True
            )

    fabric.barrier()

    train(
        fabric,
        state.model,
        state.optimizer,
        lr_scheduler,
        state.train_dataloader,
        val_dataloader,
        accu_grad_batches,
        ckpt_dir,
        grad_clip_val,
        total_nb_steps,
        log_every_n_steps,
        val_check_every,
        max_val_batches,
        ckpt_every,
        state.step_count,
        detect_anomaly,
    )


if __name__ == "__main__":
    cuda_device_name = torch.cuda.get_device_name()
    has_tensor_cores = any(el in cuda_device_name for el in ["A100", "MI250", "H100"])

    if has_tensor_cores:
        torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config")
    parser.add_argument("--optim_config")

    parser.add_argument("--num_nodes", type=int)
    parser.add_argument("--global_bs", type=int)
    parser.add_argument("--train_bs", type=int)
    parser.add_argument("--val_bs", type=int)
    parser.add_argument("--grad_clip_val", type=float)  # TODO: make a default

    parser.add_argument("--dataset")

    parser.add_argument("--run_name")
    parser.add_argument("--logdir", default="fabric_logs")

    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--strategy", default="ddp")

    # TP arguments. Default = `auto` setting in Fabric
    parser.add_argument("--dp_size", type=int, default=0)
    parser.add_argument("--tp_size", type=int, default=0)

    parser.add_argument("--ckpt_path", nargs="?", const=None, type=str)

    parser.add_argument("--training_max_seq_len", type=int, default=2048)

    parser.add_argument("--saved_ckpt_path")
    parser.add_argument("--ckpt_every", type=int, default=10000)
    parser.add_argument("--val_check_every", type=int, default=250)
    parser.add_argument("--max_val_batches", type=int, default=10)
    parser.add_argument("--log_every_n_steps", type=int, default=25)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_val_samples", type=int, default=1000)

    parser.add_argument("--load_mode", default="continue")
    parser.add_argument("--act_ckpt", action="store_true")

    parser.add_argument("--init_head", action="store_true")

    parser.add_argument("--use_profiler", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true")

    parser.add_argument("--date_in_log", action="store_true")

    args = parser.parse_args()

    main(**vars(args))
