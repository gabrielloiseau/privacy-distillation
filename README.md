# Distilling Human-Aligned Privacy Sensitivity Assessment from Large Language Models

Code and data for the paper: *Distilling Human-Aligned Privacy Sensitivity Assessment from Large Language Models*, Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi.

## Resources

All models and datasets are available in the [🤗 HuggingFace collection](https://huggingface.co/collections/gabrielloiseau/privacy-distillation).

## Quick Start

### Inference with 🤗 Transformers

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="gabrielloiseau/ettin-encoder-150m-privacy")
result = classifier("Happy First Day of Spring!")
print(result)  # [{'label': '1', 'score': 0.98}]
```

### Installation

```bash
git clone https://github.com/gabrielloiseau/privacy-distillation.git
cd privacy-distillation
uv sync
```

### Create the Dataset

To reproduce the 200k-text privacy-annotated dataset from the paper. This samples 20,000 texts from each of 10 source domains and annotates them with Mistral Large 3:

```bash
# Set your API key
export MISTRAL_API_KEY=your-key-here

# Annotate all 10 domains (supports resume)
uv run python create_dataset.py

# Annotate a single domain
uv run python create_dataset.py --domains BAC EE

# Push finished dataset to HuggingFace Hub
uv run python create_dataset.py --push --repo-id your-username/your-dataset
```

> **Note:** The Mental Health Blog (MHB) dataset is not hosted on HuggingFace. Place a parquet file with a `text` column at `data/MHB/raw_texts.parquet` before running.

### Train a Model

Fine-tune an encoder on the privacy dataset:

```bash
uv run python model_training.py
```

Override base model or hyperparameters:

```bash
uv run python model_training.py --model jhu-clsp/ettin-encoder-17m --epochs 3 --output-dir outputs/ettin-17m
```

### Evaluate on Human Benchmark

```bash
uv run python evaluate_model.py --hf-model gabrielloiseau/ettin-encoder-150m-privacy
```

### Run All Paper Experiments

```bash
uv run python run_experiments.py
```

### TAB De-identification Experiment (standalone)

```bash
uv run python tab_exp.py
```

## Repository Structure

```
privacy-distillation/
├── pyproject.toml          # Project metadata and dependencies (uv)
├── create_dataset.py       # Sample source texts + LLM annotation pipeline
├── annotator.py            # DSPy signature for teacher model annotation
├── model_training.py       # Fine-tune encoder models for privacy classification
├── evaluate_model.py       # Evaluate model against human/LLM baselines
├── run_experiments.py      # Run all paper experiments
├── tab_exp.py              # Standalone TAB de-identification experiment
├── utils.py                # Shared utilities (scoring, masking)
└── survey_data/            # 250-text human benchmark data
    ├── texts.json           # Benchmark texts
    ├── text_idx.json        # Dataset-to-index mapping
    ├── survey_results.csv   # 677 human annotations
    └── llm_improved_combined.csv  # LLM ratings
```

The survey data originates from the [privacy-judge](https://github.com/sjmeis/privacy-judge) repository, associated with the HAIPS 2025 paper *"LLM-as-a-Judge for Privacy Evaluation? Exploring the Alignment of Human and LLM Perceptions of Privacy in Textual Data"*.

## Citation

```bibtex
@misc{loiseau2026distilling,
      title={Distilling Human-Aligned Privacy Sensitivity Assessment from Large Language Models}, 
      author={Gabriel Loiseau and Damien Sileo and Damien Riquet and Maxime Meyer and Marc Tommasi},
      year={2026},
      eprint={2603.29497},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.29497}, 
}
```

