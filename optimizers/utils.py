import torch.optim as O


def get_params_with_key(model_params, keys):
    params_match = [p for n, p in model_params if any(key in n for key in keys)]
    params_no_match = [p for n, p in model_params if not any(key in n for key in keys)]

    return params_match, params_no_match


def init_optimizer(
    model_params,
    optim_cls,
    learning_rate,
    weight_decay,
    optim_kwargs,
    params_no_wd=None,
    **kwargs
):
    optim_cls = getattr(O, optim_cls)

    param_groups = None

    if params_no_wd is not None:
        no_wd_param_group, wd_param_group = get_params_with_key(
            model_params, params_no_wd
        )
        param_groups = [
            {"params": no_wd_param_group, "weight_decay": 0.0},
            {"params": wd_param_group},
        ]
    else:
        param_groups = [p for n, p in model_params]

    return optim_cls(
        param_groups, lr=learning_rate, weight_decay=weight_decay, **optim_kwargs
    )
