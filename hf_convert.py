import argparse
import json
import os

import torch
import torch.nn as nn
from lightning import Fabric
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
from lightning.fabric.plugins.precision import FSDPPrecision
from lightning.fabric.utilities.data import AttributeDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (AutoTokenizer, GPTNeoXForCausalLM, LlamaForCausalLM,
                          Olmo2ForCausalLM, TextStreamer)

from models import GPTNeoXLayer, LlamaDecoderLayer, Olmo2DecoderLayer
from models.language_model import init_model
from strategies.fsdp import FSDPStrategy

T_models = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "Olmo2ForCausalLM": Olmo2ForCausalLM,
    "GptNeoXForCausalLM": GPTNeoXForCausalLM,
}


class WrapModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lm_model = model


def main(
    ckpt_path,
    save_path,
    model_config,
    num_nodes,
    cpu_offload,
    rms_factor,
    hf_cls,
    hf_tokenizer,
):

    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer, add_bos_token=True)

    if "SLURM_NTASKS" in os.environ:
        plugins = [SLURMEnvironment()]

    layers = {Olmo2DecoderLayer, LlamaDecoderLayer, GPTNeoXLayer}

    strategy = FSDPStrategy(
        auto_wrap_policy=layers,
        precision=FSDPPrecision("bf16-true"),
        sharding_strategy="FULL_SHARD",
        use_orig_params=True,
    )

    fabric = Fabric(
        num_nodes=num_nodes, precision="bf16-true", strategy=strategy, plugins=plugins
    )
    fabric.launch()

    model_config = json.load(open(model_config, "rb"))

    with fabric.init_module(empty_init=True):
        model = init_model(**model_config)

    model = fabric.setup(model)

    state = AttributeDict(model=model)

    fabric.load(ckpt_path, state, strict=False)
    fabric.barrier()

    with FSDP.summon_full_params(
        model, writeback=False, rank0_only=True, offload_to_cpu=cpu_offload
    ):
        state_dict = model.lm_model.state_dict()
        
        tokenizer
        test_tokens = tokenizer(
            "La théorie de l'évolution est", return_tensors="pt"
        ).to("cuda")

        if fabric.is_global_zero:
            if rms_factor != 1:
                for k, v in state_dict.items():
                    if "norm" in k:
                        state_dict[k] = v * rms_factor

            hf_cls = T_models[hf_cls]
            hf_model = hf_cls.from_pretrained(
                None,
                config=model.lm_model.config,
                state_dict=state_dict,
                torch_dtype=model.lm_model.config.torch_dtype,
            )

            hf_model = hf_model.to("cuda")

            print("Testing HF model...")
            hf_model.eval()

            with torch.no_grad():
                hf_model.generate(
                    **test_tokens,
                    max_new_tokens=200,
                    do_sample=False,
                    streamer=TextStreamer(tokenizer)
                )

            hf_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path")
    parser.add_argument("--save_path")
    parser.add_argument("--model_config")
    parser.add_argument("--rms_factor", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--hf_cls", default="LlamaForCausalLM")
    parser.add_argument("--hf_tokenizer", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--cpu_offload", action="store_true")

    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    save_path = args.save_path
    rms_factor = args.rms_factor
    model_config = args.model_config
    hf_cls = args.hf_cls
    hf_tokenizer = args.hf_tokenizer
    num_nodes = args.num_nodes
    cpu_offload = args.cpu_offload

    main(
        ckpt_path,
        save_path,
        model_config,
        num_nodes,
        cpu_offload,
        rms_factor,
        hf_cls,
        hf_tokenizer,
    )
