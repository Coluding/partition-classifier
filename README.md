# Functional Diversity Classifier

Train a DeBERTa-v3-large model to classify whether two LLM-generated responses belong to the same functional partition.

Based on: *"Jointly Reinforcing Diversity and Quality in LLM Generation"* (arXiv:2509.02534)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Training

```bash
python main.py
```

### Training with Custom Parameters

```bash
python main.py \
  --data-path data/scores.jsonl \
  --num-epochs 5 \
  --train-bs 16 \
  --lr 2e-5 \
  --max-length 1024 \
  --use-wandb \
  --output-dir ./my-model
```

### View All Options

```bash
python main.py --help
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | `data/scores.jsonl` | Path to training data |
| `--num-epochs` | `3` | Number of training epochs |
| `--train-bs` | `8` | Training batch size |
| `--eval-bs` | `4` | Evaluation batch size |
| `--lr` | `1e-5` | Learning rate |
| `--max-length` | `1024` | Max sequence length |
| `--use-wandb` | `False` | Enable W&B logging |
| `--output-dir` | `./deberta-functional-diversity` | Output directory |

## Data Format

Input JSONL file where each line contains:

```json
{
  "id": "unique_identifier",
  "prompt": "the original prompt",
  "generations": ["response 1", "response 2", ...],
  "partition": [0, 0, 1, 1, 2, ...]
}
```

The `partition` array indicates which responses have the same functional behavior (same ID = same partition).

## Output

Training produces:
- Model checkpoints in `output_dir/`
- Training logs in `output_dir/training.log`
- TensorBoard logs in `logging_dir/`
- Prediction examples after each epoch (logged to console, file, TensorBoard, and W&B)

### View TensorBoard Logs

```bash
tensorboard --logdir ./logs
```

## Example Run

```bash
# Train for 5 epochs with larger batch size and W&B logging
python main.py \
  --num-epochs 5 \
  --train-bs 16 \
  --eval-bs 8 \
  --lr 2e-5 \
  --use-wandb \
  --run-name "deberta-large-5ep"
```