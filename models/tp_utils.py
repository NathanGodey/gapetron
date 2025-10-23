from torch.distributed._composable.fsdp.fully_shard import fully_shard
# from torch.distributed.tensor.placement_types import Shard, Replicate
from torch.distributed._tensor import Replicate, Shard
# torch.distributed._tensor
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               PrepareModuleInput,
                                               RowwiseParallel,
                                               SequenceParallel,
                                               parallelize_module)

from models import LlamaForCausalLM


def parallelize_llama(model, tp_mesh):
    # TODO: embeddings only work in Colwise parallel, why?
    model = parallelize_module(
        model,
        tp_mesh,
        {
            "model.embed_tokens": ColwiseParallel(
                input_layouts=Replicate(), output_layouts=Shard(1)
            ),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1), output_layouts=Replicate()
            ),
            "model.norm": SequenceParallel(),
        },
    )

    for layer in model.model.layers:
        parallelize_module(
            layer,
            tp_mesh,
            {
                "input_layernorm": SequenceParallel(),
                "post_attention_layernorm": SequenceParallel(),
                "self_attn": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "mlp": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "self_attn.q_proj": ColwiseParallel(),
                "self_attn.k_proj": ColwiseParallel(),
                "self_attn.v_proj": ColwiseParallel(),
                "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                "mlp.gate_proj": ColwiseParallel(),
                "mlp.up_proj": ColwiseParallel(),
                "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            },
        )

    return model


def parallelize_model(model, device_mesh):
    
    tp_mesh = device_mesh["tensor_parallel"]
    dp_mesh = device_mesh["data_parallel"]

    if tp_mesh.size() > 1:
        if isinstance(model.lm_model, LlamaForCausalLM):
            model.lm_model = parallelize_llama(model.lm_model, tp_mesh)
        else:
            raise ValueError(
                f"Model class {type(model.lm_model)} has no TP plan implemented."
            )

    if dp_mesh.size() > 1:
        fully_shard(model, mesh=dp_mesh)
    return model
