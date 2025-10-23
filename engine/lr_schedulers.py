from torch.optim.lr_scheduler import LambdaLR
from transformers import get_constant_schedule_with_warmup
from transformers.optimization import \
    get_cosine_with_min_lr_schedule_with_warmup


def get_linear_with_min_lr_schedule_with_warmup(
    optimizer, warmup_steps, total_steps, min_lr_rate
):
    if warmup_steps > total_steps:
        raise ValueError(
            f"Provided larger warmup ({warmup_steps}) than total steps ({total_steps}) in LR scheduler."
        )
    assert 0 <= min_lr_rate <= 1

    def _linear_sched_func(i):
        if i < warmup_steps:
            return i / warmup_steps
        else:
            step_ratio = 1 - (i - warmup_steps) / (total_steps - warmup_steps)
            return step_ratio + (1 - step_ratio) * min_lr_rate

    return LambdaLR(optimizer, _linear_sched_func)


def get_wsd_schedule_with_warmup(
    optimizer, warmup_steps, total_steps, cooldown_steps, min_lr_rate
):
    if warmup_steps > total_steps:
        raise ValueError(
            f"Provided larger warmup ({warmup_steps}) than total steps ({total_steps}) in LR scheduler."
        )
    if cooldown_steps > total_steps:
        raise ValueError(
            f"Provided larger warmup ({cooldown_steps}) than total steps ({total_steps}) in LR scheduler."
        )
    if cooldown_steps + warmup_steps > total_steps:
        raise ValueError(
            f"Provided larger warmup+cooldown ({warmup_steps}+{cooldown_steps}) than total steps ({total_steps}) in LR scheduler."
        )
    assert 0 < min_lr_rate <= 1

    plateau_steps = total_steps - cooldown_steps

    def _wsd_sched_func(i):
        if i < warmup_steps:
            return i / warmup_steps
        elif warmup_steps <= i < plateau_steps:
            return 1
        else:
            step_ratio = 1 - (i - plateau_steps) / (total_steps - plateau_steps)
            return step_ratio + (1 - step_ratio) * min_lr_rate

    return LambdaLR(optimizer, _wsd_sched_func)


def init_lr_scheduler(
    optimizer,
    schedule_type,
    warmup_steps=None,
    total_nb_steps=None,
    cooldown_steps=None,
    min_lr_rate=0.1,
    **kwargs,
):
    if schedule_type == "cosine":
        lr_scheduler_func = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer, warmup_steps, total_nb_steps, min_lr_rate=min_lr_rate
        )
    elif schedule_type == "constant":
        lr_scheduler_func = get_constant_schedule_with_warmup(
            optimizer,
            warmup_steps,
        )
    elif schedule_type == "linear":
        lr_scheduler_func = get_linear_with_min_lr_schedule_with_warmup(
            optimizer, warmup_steps, total_nb_steps, min_lr_rate=min_lr_rate
        )
    elif schedule_type == "wsd":
        lr_scheduler_func = get_wsd_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_nb_steps,
            cooldown_steps,
            min_lr_rate=min_lr_rate,
        )
    else:
        raise NotImplementedError(
            f"LR schedule named '{schedule_type}' is not implemented."
        )
    return lr_scheduler_func
