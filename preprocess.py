import argparse
import importlib.util
import json
import os
import random
import sys
import warnings
from functools import partial, wraps
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from smart_open import open
from transformers import AutoTokenizer

from litdata import TokensLoader, optimize


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = warning_on_one_line


parser = argparse.ArgumentParser()
parser.add_argument("--num_proc")
parser.add_argument("--batch_size")
parser.add_argument("--max_seq_len")
parser.add_argument("--reduplicate", action="store_true")
parser.add_argument("--process", action="store_true")


parser.add_argument("--node_id", default=0)
parser.add_argument("--num_nodes", default=1)

parser.add_argument("--chunk_size", default="64MB")

parser.add_argument("--tokenizer")

parser.add_argument("--dataset_path")
parser.add_argument("--dataset_config", default="default")
parser.add_argument("--filename_filter", default="")

parser.add_argument("--output_path")


args = parser.parse_args()

num_proc = int(args.num_proc)
max_seq_len = int(args.max_seq_len)
batch_size = int(args.batch_size)
node_id = int(args.node_id)
num_nodes = int(args.num_nodes)
tokenizer = args.tokenizer
dataset_path = args.dataset_path
filename_filter = args.filename_filter

dataset_config = args.dataset_config
output_path = args.output_path
chunk_size = args.chunk_size
reduplicate = args.reduplicate
process = args.process


if reduplicate:
    try:
        spec = importlib.util.spec_from_file_location(
            "redup_module", f"{dataset_path}/reduplication.py"
        )
        redup_module = importlib.util.module_from_spec(spec)
        sys.modules["redup_module"] = redup_module
        spec.loader.exec_module(redup_module)
        assert hasattr(redup_module, "reduplicate_fn")
    except (FileNotFoundError, AssertionError):
        raise ValueError(
            f"Missing reduplication.py file in {dataset_path}. A reduplicate_fn needs to be provided when --reduplicate is used."
        )

if process:
    try:
        spec = importlib.util.spec_from_file_location(
            "proc_module", f"{dataset_path}/processing.py"
        )
        proc_module = importlib.util.module_from_spec(spec)
        sys.modules["proc_module"] = proc_module
        spec.loader.exec_module(proc_module)
        assert hasattr(proc_module, "process_fn")
    except (FileNotFoundError, AssertionError):
        raise ValueError(
            f"Missing processing.py file in {dataset_path}. A process_fn needs to be provided when --process is used."
        )


if num_nodes > 1:
    output_path = output_path + f"/node_{node_id}"

tokenizer = AutoTokenizer.from_pretrained(tokenizer)


def safe_encode(text, tokenizer, max_tok_len=1_000_000):
    if len(text) < max_tok_len:
        return tokenizer.encode(text)
    tokenized_chunks = [
        tokenizer.encode(
            text[i * max_tok_len : (i + 1) * max_tok_len], add_special_tokens=(i == 0)
        )
        for i in range(len(text) // max_tok_len + 1)
    ]
    flat_token_seq = sum(tokenized_chunks, [])
    return flat_token_seq


def yield_and_reduplicate(sample_tensor, data, filepath):
    if reduplicate:
        reduplicate_count = redup_module.reduplicate_fn(data, filepath)
        for _ in range(reduplicate_count):
            yield sample_tensor
    else:
        yield sample_tensor


def shuffle_yield(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        random.shuffle(result)
        for item in result:
            yield item

    return wrapper


# 1. Define a function to convert the text within the parquet files into tokens
@shuffle_yield
def tokenize_parquet_fn(filepath, tokenizer=None, batch_size=8192):
    parquet_file = pq.ParquetFile(filepath)
    samples_to_yield = []
    # Process per batch to reduce RAM usage
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for _, data in batch.to_pandas().iterrows():
            # tokens = tokenizer.encode(text)
            if process:
                data["text"] = proc_module.process_fn(data)
            tokens = safe_encode(data["text"], tokenizer)
            sample_tensor = torch.tensor(tokens + [tokenizer.eos_token_id])
            samples_to_yield.append(sample_tensor)

            if reduplicate:
                reduplicate_count = redup_module.reduplicate_fn(data, filepath)
                for _ in range(reduplicate_count):
                    samples_to_yield.append(sample_tensor)
            else:
                samples_to_yield.append(sample_tensor)
    return samples_to_yield


@shuffle_yield
def tokenize_arrow_fn(filepath, tokenizer=None, batch_size=8192):
    # TODO: unfinished function
    with pa.memory_map(filepath, "rb") as source:
        loaded_array = pa.ipc.open_file(source).read_all()
        for text in loaded_array["text"]:
            if process:
                text = proc_module.process_fn(text)
            tokens = safe_encode(text, tokenizer)
            yield torch.tensor(tokens + [tokenizer.eos_token_id])


@shuffle_yield
def tokenize_jsongz_fn(filepath, tokenizer=None, batch_size=None):
    samples_to_yield = []
    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line if isinstance(line, str) else line.decode("utf-8"))
            if process:
                data["text"] = proc_module.process_fn(data)
                for sample in data["text"]:
                    tokens = safe_encode(sample, tokenizer)
                    sample_tensor = torch.tensor(tokens + [tokenizer.eos_token_id])
                    samples_to_yield.append(sample_tensor)

                    if reduplicate:
                        reduplicate_count = redup_module.reduplicate_fn(data, filepath)
                        for _ in range(reduplicate_count):
                            samples_to_yield.append(sample_tensor)
                    else:
                        samples_to_yield.append(sample_tensor)
                continue

            tokens = safe_encode(data["text"], tokenizer)
            sample_tensor = torch.tensor(tokens + [tokenizer.eos_token_id])

            if reduplicate:
                reduplicate_count = redup_module.reduplicate_fn(data, filepath)
                for _ in range(reduplicate_count):
                    samples_to_yield.append(sample_tensor)
            else:
                samples_to_yield.append(sample_tensor)
    return samples_to_yield


@shuffle_yield
def tokenize_zst_fn(filepath, tokenizer=None, batch_size=None):
    samples_to_yield = []
    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line if isinstance(line, str) else line.decode("utf-8"))
            if process:
                data["text"] = proc_module.process_fn(data)
            tokens = safe_encode(data["text"], tokenizer)
            sample_tensor = torch.tensor(tokens + [tokenizer.eos_token_id])
            samples_to_yield.append(sample_tensor)

            if reduplicate:
                reduplicate_count = redup_module.reduplicate_fn(data, filepath)
                for _ in range(reduplicate_count):
                    samples_to_yield.append(sample_tensor)
            else:
                samples_to_yield.append(sample_tensor)
    return samples_to_yield


# 2. Generate the inputs

tokenize_fn = tokenize_parquet_fn
inputs = [str(file) for file in Path(dataset_path).rglob("*.parquet")]

if len(inputs) == 0:
    tokenize_fn = tokenize_arrow_fn
    inputs = [str(file) for file in Path(dataset_path).rglob("*.arrow")]

if len(inputs) == 0:
    tokenize_fn = tokenize_jsongz_fn
    inputs = [str(file) for file in Path(dataset_path).rglob("*.jsonl.gz")]
    inputs.extend([str(file) for file in Path(dataset_path).rglob("*.jsonl")])
    inputs.extend([str(file) for file in Path(dataset_path).rglob("*.json.gz")])
    inputs.extend([str(file) for file in Path(dataset_path).rglob("*.json")])


if len(inputs) == 0:
    tokenize_fn = tokenize_zst_fn
    inputs = [str(file) for file in Path(dataset_path).rglob("*.zst")]

if len(inputs) == 0:
    raise FileNotFoundError(
        "Could not find any .parquet, .arrow, .json(l).gz or .json(l) files in provided directory."
    )

inputs_non_empty = [
    input_file for input_file in inputs if os.stat(input_file).st_size > 0
]
if len(inputs_non_empty) < len(inputs):
    print(
        f"{len(inputs) - len(inputs_non_empty)} file(s) are empty and will be ignored"
    )
    inputs = inputs_non_empty

if filename_filter:
    inputs = [fname for fname in inputs if filename_filter in fname]

inputs = [fname for i, fname in enumerate(inputs) if i % num_nodes == node_id]


if num_proc > len(inputs):
    warnings.warn(
        f"Setting num_proc to {len(inputs)} as it is the number of files for this job.",
        UserWarning,
    )
    num_proc = len(inputs)


if __name__ == "__main__":
    # 3. Store the optimized data wherever you want under "/teamspace/datasets" or "/teamspace/s3_connections"
    print(f"Node {node_id} processing {len(inputs)} files: \n" + "\n".join(inputs))
    outputs = optimize(
        fn=partial(
            tokenize_fn, tokenizer=tokenizer, batch_size=batch_size
        ),  # Note: Use HF tokenizer or any others
        inputs=inputs,
        num_workers=num_proc if num_proc > 0 else None,
        output_dir=output_path,
        chunk_bytes=chunk_size,  # Number of tokens to store by chunks. This is roughly 64MB of tokens per chunk.
        item_loader=TokensLoader(),
    )
