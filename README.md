# Gapetron

![License](https://img.shields.io/github/license/NathanGodey/gapetron)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

Gapetron is a modular, hackable toolkit for large-scale pretraining of language models (LLMs). It was used to train the Gaperon LLM suite and is designed for researchers and engineers who want a flexible, minimal, and easily extensible codebase for LLM experimentation and production.

## Features
### Parallelism & Scaling

- **DDP (Distributed Data Parallel)**
- **FSDP (Fully Sharded Data Parallel)** 
  - Full shard
  - Gradient operation sharding (`fsdp_grad_op`)
  - Hybrid sharding (`fsdp_hybrid`)
- **Tensor Parallelism (TP):** Model parallelism across GPUs (set `tp_size`).
- **Custom Strategies:** Easily add or swap new strategies in `strategies/`.

### Optimization

- **Optimizer Choice:** Select any PyTorch optimizer (Adam, AdamW, Muon, RMSProp, ...) via config.
- **Learning Rate Scheduling:** Constant/step, cosine, linear warmup/cooldown, WSD; editable via config.
- **Gradient Accumulation \& Clipping**
- **Activation Checkpointing**
- **Precision Control:** Mixed/Pure bfloat16/float16, float32 (`precision` argument).

### Data Handling

- **Dataset Mixes:** Train on any combination of datasets (configurable in data JSONs).
- **Streaming/Offline Data:** Supports HuggingFace streaming and fully offline modes (env vars).
- **Instant Checkpoint Restart:** Resume training from checkpoints—full state, model-only, optimizer-only, transfer, cooldown, etc.
- **Custom Preprocessing:** Use or extend `preprocess.py` for custom data pipelines.

### Model & Attention

- **Direct Compatibility with HuggingFace AutoModels**.
- **Model Family:** Choose Llama3, Olmo2, GPTNeoX, or add your own in `models/`.
- **Attention Implementations:** FlashAttention 2 \& 3.
- **Torch compilation**

### Experiment Management & Logging

- **Flexible Logging:** TensorBoard logging, custom run names, log directories.
- **Profiler & Debug:** Optional PyTorch-based profiling, anomaly detection.
- **Custom Checkpointing \& Validation:** Long-term and short-term frequency, validation epoch frequency, validation sample size, —all adjustable.

### Cluster & HPC Support

- **SLURM Integration:** Out-of-the-box support for SLURM-based HPC clusters.
- **Seamless walltime handling:** Allows for SLURM auto-restarts with last checkpoint loading.

### Hardware

- **Tested on Nvidia \& AMD GPUs across different generations:**
   - AMD MI250
   - AMD MI300
   - Nvidia Ampere (A100)
   - Nvidia Hopper (H100)


## Repository Structure

- `configs/` – Configuration files for training runs, models, and datasets.
- `data/` – Utilities and scripts for data preprocessing and management.
- `engine/` – Core training engine logic.
- `models/` – Model architectures and building blocks.
- `optimizers/` – Optimizer definitions for various training regimes.
- `strategies/` – Distributed and parallel training strategies.
- `scripts/` – Entry points and helper scripts for launching jobs and managing experiments.
- `preprocess.py` – Script for preparing raw datasets for training.
- `train.py` – Main script to launch LLM pretraining.
- `merge_ds.py` – Utility for merging multiple datasets.
- `hf_convert.py` – Tools for converting models to/from Hugging Face format.

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/NathanGodey/gapetron.git
   cd gapetron
   ```

2. **Set up your environment**

   Install required dependencies (see your preferred config or requirements file).

3. **Prepare your data**

   Use `preprocess.py` to process raw text datasets into the required format.

   ```bash
   python preprocess.py --input your_raw_data.txt --output data/processed/
   ```

   Optionally, merge datasets with `merge_ds.py`.

4. **Configure your experiment**

   Edit or create a config file in the `configs/` directory to specify model size, training parameters, etc.

5. **Train your model**

   Launch training via:

   ```bash
   python train.py --config configs/your_config.yaml
   ```

6. **Advanced**

   - Customize models by editing files in `models/`
   - Add new optimizers in `optimizers/`
   - Implement new distributed training tricks in `strategies/`
   - Convert checkpoints to Hugging Face format with `hf_convert.py`

## Hackability

Gapetron is intentionally designed for rapid prototyping and experimentation:

- Minimal boilerplate and clear code flow
- Self-contained, short Python scripts
- Modular directory structure—swap in your own components easily

## License

Apache License 2.0

---

Gapetron is an independent project inspired by practical needs in training large language models. Contributions and issues are welcome!
