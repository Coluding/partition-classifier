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

## Example Runs

### Full Fine-Tuning
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

### LoRA Training (Parameter-Efficient)

For memory-efficient training, use LoRA (Low-Rank Adaptation):

```bash
# First install peft: pip install peft

# Train with LoRA - much lower memory usage
python main.py \
  --use-lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --num-epochs 5 \
  --train-bs 32 \
  --lr 3e-4
```

**LoRA Benefits:**
- Trains only 0.1-1% of model parameters
- Significantly reduced memory usage
- Faster training
- Can use larger batch sizes

## Loading Trained Models

### Loading Full Fine-Tuned Models

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("./deberta-functional-diversity/best_model")
tokenizer = AutoTokenizer.from_pretrained("./deberta-functional-diversity/best_model")
```

Or using the utility script:
```bash
python load_model.py ./deberta-functional-diversity/best_model --test
```

### Loading LoRA Fine-Tuned Models

LoRA models require the base model + adapter:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-large",
    num_labels=2
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./deberta-functional-diversity/best_model")

# Merge for deployment (optional)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
```

Or using the utility script:
```bash
python load_model.py \
  ./deberta-functional-diversity/best_model \
  --lora \
  --base-model microsoft/deberta-v3-large \
  --test
```

## Interactive Dashboard

Launch a web interface to test your model interactively:

### For Full Fine-Tuned Models
```bash
python dashboard.py ./deberta-functional-diversity/best_model
```

### For LoRA Models
```bash
python dashboard.py \
  ./deberta-functional-diversity/best_model \
  --lora \
  --base-model microsoft/deberta-v3-large
```

### Dashboard Options
- `--share`: Create a public shareable link (accessible from anywhere)
- `--port 7860`: Specify custom port (default: 7860)
- `--device cpu`: Force CPU usage

The dashboard will open in your browser at `http://localhost:7860` with:
- Text input fields for two responses
- Real-time predictions with confidence scores
- Probability breakdowns
- Pre-loaded example pairs