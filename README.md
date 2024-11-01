# Selective Concept Removal Framework for LLMs

## Overview
This project provides a framework for selectively removing specific concepts from the training of large language models (LLMs). The implementation leverages LoRA (Low-Rank Adaptation) to inject specialized parameters into specific layers of a pre-trained LLaMA model, enabling the modification of model behavior without retraining from scratch.


## Usage

### Training the Model
To train the concept removal model, use the `run_training.py` script. Example usage:

```sh
python scripts/run_training.py --data_path path/to/data --concept_path path/to/concepts --base_model_path path/to/base/model
```

Arguments:
- `--data_path`: Path to the training dataset.
- `--concept_path`: Path to the JSON file defining the concepts.
- `--base_model_path`: Path to the pre-trained LLaMA model.

### Evaluating the Model
To evaluate the trained model, use the `run_evaluation.py` script. Example usage:

```sh
python scripts/run_evaluation.py --data_path path/to/evaluation/data --concept_path path/to/concepts --model_path path/to/model/checkpoint --base_model_path path/to/base/model
```

Arguments:
- `--data_path`: Path to the evaluation dataset.
- `--concept_path`: Path to the JSON file defining the concepts.
- `--model_path`: Path to the saved model checkpoint.
- `--base_model_path`: Path to the pre-trained LLaMA model.

