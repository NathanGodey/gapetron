# Gapetron

![License](https://img.shields.io/github/license/NathanGodey/gapetron)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

Gapetron is a modular, hackable toolkit for large-scale pretraining of language models (LLMs). It was used to train the Gaperon LLM suite and is designed for researchers and engineers who want a flexible, minimal, and easily extensible codebase for LLM experimentation and production.

## Features

- **Short and Modular**: The codebase is intentionally kept compact and organized, making it easy to grasp the overall flow and quickly modify or extend key components.
- **Configurable**: All major components (models, optimizers, training strategies) are separated, with flexible configs to support various research or production-scale setups.
- **Hackable**: The code is written with clarity and modularity in mind, enabling fast iteration and experimentation.
- **Production-Ready**: Used to train the Gaperon LLM suite at scale.

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
