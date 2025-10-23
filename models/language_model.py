import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

import models as M
from engine.loss_fns import cwt_loss


class LmWithLoss(nn.Module):
    def __init__(self, lm_model, is_headless):
        super().__init__()
        self.lm_model = lm_model
        self.is_headless = is_headless

    def get_input_embeddings(self):
        return self.lm_model.get_input_embeddings()

    def to_hf(self):
        return self.lm_model.to_hf()

    def init_weights(self):
        self.lm_model.apply(self.lm_model._init_weights)

    def forward(self, inputs, output_preds=True):
        labels = inputs[..., 1:]
        lm_result = self.lm_model(inputs[..., :-1])
        logits = lm_result.logits
        lm_predictions = None

        if self.is_headless:
            target_input_embeddings = self.lm_model.model._input_embs(labels).flatten(
                0, 1
            )
            target_input_embeddings = target_input_embeddings.to(logits.dtype)
            lm_loss = cwt_loss(
                logits.flatten(0, 1),
                positive=target_input_embeddings,
                negative=target_input_embeddings,
            )
        else:
            # print(logits, labels)
            # TODO: check if need to call .float()
            lm_loss = F.cross_entropy(logits.transpose(1, 2), labels)
            if output_preds:
                lm_predictions = logits.detach().argmax(-1)

        return lm_loss, lm_predictions, labels


def init_model(
    hf_path,
    hf_load_weights,
    attn_type,
    is_headless,
    model_cls,
    vocab_size,
    max_position_embeddings,
    torch_compile,
    torch_compile_mode,
    hf_overwrite=None,
    **kwargs
):
    model_cls = getattr(M, model_cls)
    if hf_load_weights:
        hf_model = model_cls.hf_model_cls.from_pretrained(
            hf_path, attn_implementation=attn_type, use_cache=False
        )
        lm_model = model_cls.from_hf(hf_model)
    else:
        lm_config = AutoConfig.from_pretrained(hf_path, use_cache=False)
        if hf_overwrite is not None:
            for key, value in hf_overwrite.items():
                setattr(lm_config, key, value)
        lm_config.vocab_size = vocab_size
        lm_config.max_position_embeddings = max_position_embeddings
        lm_config._attn_implementation = attn_type

        lm_model = model_cls(lm_config)

        if is_headless:
            lm_model.lm_head = torch.nn.Identity()

    lm_model = LmWithLoss(lm_model, is_headless)

    if torch_compile:
        return torch.compile(lm_model, mode=torch_compile_mode)

    return lm_model
