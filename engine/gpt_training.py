# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import signal
import subprocess
import sys
from functools import partial

import lightning as L
import numpy as np
import torch
from lightning.fabric.wrappers import _FabricDataLoader
from tqdm import tqdm

from data.utils import CycleIterator


def print_rank_zero(*args, fabric, **kwargs):
    if fabric.is_global_zero:
        print(*args, **kwargs)


# def handle_requeue(*args, fabric, state, ckpt_dir):
#     fabric.barrier()

#     ckpt_total_path = f"{ckpt_dir}/requeue.ckpt"
#     fabric.save(ckpt_total_path, state)

#     if fabric.is_global_zero:
#         job_id = os.environ.get("SLURM_JOB_ID")
#         print(f"Attempting to requeue job {job_id}...", flush=True)
#         subprocess.run(["scontrol", "requeue", job_id])

#     fabric.barrier()


def save_and_requeue(fabric, state, ckpt_dir):
    if fabric.is_global_zero:
        print("Saving requeue checkpoint...")
    fabric.barrier()

    ckpt_total_path = f"{ckpt_dir}/requeue.ckpt"
    fabric.save(ckpt_total_path, state)

    if fabric.is_global_zero:
        job_id = os.environ.get("SLURM_JOB_ID")
        print(f"Requeuing job {job_id}...", flush=True)
        subprocess.run(["scontrol", "requeue", job_id])

    fabric.barrier()
    sys.exit()


def get_prog_bar(iterable, fabric, **kwargs):
    if fabric.is_global_zero:
        return tqdm(iterable, **kwargs)
    return iterable


def train(
    fabric: L.Fabric,
    model,
    optimizer,
    lr_scheduler,
    train_dataloader: _FabricDataLoader,
    val_dataloader: _FabricDataLoader,
    grad_accum_steps,
    ckpt_dir,
    grad_clip_val,
    max_iters,
    log_interval,
    eval_interval,
    max_val_batches,
    ckpt_every,
    offset_step_count=0,
    detect_anomaly=False,
):
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """

    fabric_print_rank_zero = partial(print_rank_zero, fabric=fabric)

    fabric_print_rank_zero(
        f"Validation Dataloader State_dict Before : {str(val_dataloader._dataloader._num_samples_yielded_combined)}"
    )

    step_count = offset_step_count

    global walltime_signal_received
    walltime_signal_received = False

    def handle_requeue(*args):
        global walltime_signal_received
        fabric_print_rank_zero(
            "Walltime signal received. Saving requeue checkpoint after next optimization step."
        )
        walltime_signal_received = True

    signal.signal(signal.SIGUSR1, handle_requeue)

    lr = lr_scheduler.get_last_lr()[0]
    with torch.autograd.set_detect_anomaly(detect_anomaly):
        for iter_num, train_batch in get_prog_bar(
            enumerate(CycleIterator(train_dataloader)),
            fabric,
            desc="Training",
            total=max_iters,
            initial=step_count,
        ):
            # determine and set the learning rate for this iteration

            input_ids = train_batch.contiguous()

            is_accumulating = (iter_num + 1) % grad_accum_steps != 0

            with fabric.no_backward_sync(model, enabled=is_accumulating):
                # print(os.environ.get("SLURMD_NODENAME"), fabric.global_rank, input_ids)
                lm_loss, lm_predictions, labels = model(
                    input_ids, output_preds=(not is_accumulating)
                )
                fabric.backward(lm_loss / grad_accum_steps)

            if not is_accumulating:

                if step_count == offset_step_count:
                    if fabric.is_global_zero:
                        log_stats(
                            fabric,
                            labels,
                            lm_predictions,
                            lm_loss,
                            train_dataloader._dataloader.state_dict(),
                            lr=lr,
                            is_headless=model.is_headless,
                            step_count=step_count,
                        )
                    fabric.barrier()

                fabric.clip_gradients(
                    model, optimizer, max_norm=grad_clip_val, error_if_nonfinite=False
                )

                optimizer.step()
                lr_scheduler.step()
                lr = lr_scheduler.get_last_lr()[0]
                optimizer.zero_grad()
                step_count += 1

                if step_count % log_interval == 0 and fabric.is_global_zero:

                    if fabric.is_global_zero:
                        log_stats(
                            fabric,
                            labels,
                            lm_predictions,
                            lm_loss,
                            train_dataloader._dataloader.state_dict(),
                            lr=lr,
                            is_headless=model.is_headless,
                            step_count=step_count,
                        )

                if val_dataloader is not None and step_count % eval_interval == 0:
                    fabric_print_rank_zero("Validating...", flush=True)
                    fabric_print_rank_zero(
                        f"Validation Dataloader State_dict Before : {str(val_dataloader._dataloader._num_samples_yielded_combined)}"
                    )
                    metrics_dict = validate(
                        fabric, model, val_dataloader, max_val_batches
                    )
                    fabric_print_rank_zero(
                        f"Validation Dataloader State_dict After : {str(val_dataloader._dataloader._num_samples_yielded_combined)}"
                    )
                    reduced_metrics_dict = fabric.all_reduce(metrics_dict)
                    fabric.log_dict(reduced_metrics_dict, step=step_count)

                fabric.barrier()

                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "lr_scheduler": lr_scheduler,
                    "train_dataloader": train_dataloader,
                    "step_count": step_count,
                }

                if walltime_signal_received:
                    save_and_requeue(fabric=fabric, state=state, ckpt_dir=ckpt_dir)

                # signal.signal(signal.SIGUSR1, partial(handle_requeue, fabric=fabric, state=state, ckpt_dir=ckpt_dir))

                if step_count % ckpt_every == 0:
                    ckpt_total_path = f"{ckpt_dir}/step_{step_count}.ckpt"
                    fabric_print_rank_zero(
                        f"\nSaving checkpoint to {ckpt_total_path}", flush=True
                    )
                    fabric.save(ckpt_total_path, state)
                    fabric.barrier()

                if step_count >= max_iters:
                    fabric_print_rank_zero(
                        f"Training stopped after {step_count} steps."
                    )
                    fabric.barrier()
                    break


def log_stats(
    fabric, labels, lm_predictions, lm_loss, dl_state, lr, is_headless, step_count
):
    free, total = torch.cuda.mem_get_info()
    metrics_dict = {
        "train/loss": lm_loss,
        "lr": lr,
        "hardware/vram_usage": (total - free) / total,
    }

    console_log_str = f"""Training Metrics:

    LR: {str(lr)}
    VRAM Usage: {str((total - free) / total)}

    train/loss: {str(lm_loss.item())}
    """

    if not is_headless:
        with torch.no_grad():
            lm_acc = (labels == lm_predictions).float().mean()
        metrics_dict["train/acc"] = lm_acc
        console_log_str += f"train/acc: {str(lm_acc.item())}\n"

    total_tokens = np.int64(0)
    for ds in dl_state["dataset"].values():

        ds_name = ds["input_dir_path"].split("/")[-1]
        token_count = np.int64(ds["item_loader"]["block_size"]) * np.int64(
            ds["num_samples_yielded"] * fabric.world_size
        )

        metrics_dict[f"data/{ds_name}_tokens"] = token_count
        total_tokens += token_count
        metrics_dict[f"data/{ds_name}_epochs"] = ds["current_epoch"]

        # console_log_str += f"data/{ds_name}_tokens: {str(token_count)}\n"
        # console_log_str += f"data/{ds_name}_epochs: {str(ds['current_epoch'])}\n"

    metrics_dict["data/total_tokens"] = total_tokens
    metrics_dict["data/total_epochs"] = dl_state["current_epoch"]

    console_log_str += f"    data/total_tokens: {str(total_tokens)}\n"
    console_log_str += f"    data/total_epochs: {str(dl_state['current_epoch'])}\n"

    print_rank_zero(console_log_str, fabric=fabric, flush=True)

    fabric.log_dict(metrics_dict, step=step_count)


# TODO: no grad leads to recompilation for AMD : fix?
@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: torch.nn.Module,
    val_dataloader: _FabricDataLoader,
    max_val_batches,
) -> torch.Tensor:
    losses = torch.zeros(max_val_batches)
    if not model.is_headless:
        accuracies = torch.zeros(max_val_batches)

    for k, val_batch in get_prog_bar(
        enumerate(CycleIterator(val_dataloader)),
        fabric,
        desc="Validating",
        total=max_val_batches,
    ):
        if k >= max_val_batches:
            break
        input_ids = val_batch.contiguous()
        lm_loss, lm_predictions, labels = model(input_ids, output_preds=True)
        if not model.is_headless:
            with torch.no_grad():
                lm_acc = (labels == lm_predictions).float().mean()
            accuracies[k] = lm_acc
        losses[k] = lm_loss.item()
    metrics_dict = {"val/loss": losses.mean()}
    if not model.is_headless:
        metrics_dict["val/acc"] = accuracies.mean()

    print_rank_zero(
        f"Validation Metrics:\n    val/loss: {str(losses.mean().item())}\n    val/acc: {str(accuracies.mean().item())}",
        fabric=fabric,
        flush=True,
    )

    return metrics_dict
