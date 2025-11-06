"""
Train a functional diversity classifier as in
'Jointly Reinforcing Diversity and Quality in LLM Generation' (arXiv:2509.02534).
Uses DeBERTa-v3-large fine-tuned on response pairs with custom PyTorch training loop.
"""

import os
import logging
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, List, Dict, Any
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed,
    get_linear_schedule_with_warmup,
)

from dataset import FunctionalPairDataset
from utils import setup_logger

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False



# -------------------- Configuration -------------------- #
@dataclass
class Config:
    model_name = "microsoft/deberta-v3-large"
    train_path = "data/new_dataset/train.jsonl"
    val_path = "data/new_dataset/val.jsonl"
    output_dir = "./deberta-functional-diversity"
    logging_dir = "./logs"
    seed = 42
    test_size = 0.1
    num_pairs_per_epoch = 16_000
    n_pairs_per_prompt = 10
    max_length = 1024  # maximum sequence length for tokenization
    neg_to_pos_ratio = 0.5

    # Training
    lr = 1e-5
    num_epochs = 30
    train_bs = 3
    eval_bs = 6
    weight_decay = 0.01
    warmup_steps = 0
    max_grad_norm = 1.0
    accumulation_steps = 1

    # Device
    device: str = field(init=False)
    fp16: bool = field(init=False)

    # LoRA parameters
    use_lora = False  # enable LoRA training
    lora_r = 8  # LoRA rank
    lora_alpha = 16  # LoRA alpha (scaling factor)
    lora_dropout = 0.1  # LoRA dropout
    lora_target_modules = ["query_proj", "key_proj", "value_proj", "dense"]  # modules to apply LoRA to

    # Logging
    use_wandb = True
    use_tensorboard = True
    project_name = "functional-diversity-classifier"
    run_name = f"{max_length}_{num_pairs_per_epoch}_{neg_to_pos_ratio}" + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_every_n_steps = 10
    eval_every_n_epochs = 1
    save_every_n_epochs = 1
    num_examples_to_log = 3  # number of prediction examples to log each epoch

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16 = (self.device == "cuda")


# -------------------- Example Logging -------------------- #

def log_prediction_examples(model, dataset, tokenizer, device, epoch, cfg, logger, writer=None):
    """Log example predictions with their inputs to tensorboard and wandb."""
    model.eval()

    # Get random examples from dataset
    indices = np.random.choice(len(dataset), min(cfg.num_examples_to_log, len(dataset)), replace=False)

    examples = []
    with torch.no_grad():
        for idx in indices:
            item = dataset[idx]

            # Get the raw pair data
            pair_data = dataset._pairs_to_sample[idx]
            response_a = pair_data["response_a"]
            response_b = pair_data["response_b"]
            true_label = pair_data["label"]
            prompt = pair_data.get("prompt", "N/A")

            # Get prediction
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention_mask = item["attention_mask"].unsqueeze(0).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_label = torch.argmax(logits, dim=-1).item()

            example = {
                "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "response_a": response_a[:300] + "..." if len(response_a) > 300 else response_a,
                "response_b": response_b[:300] + "..." if len(response_b) > 300 else response_b,
                "true_label": "Same Partition" if true_label == 1 else "Different Partition",
                "pred_label": "Same Partition" if pred_label == 1 else "Different Partition",
                "confidence": f"{probs[pred_label].item():.4f}",
                "correct": "✓" if pred_label == true_label else "✗",
                "prob_same": f"{probs[1].item():.4f}",
                "prob_diff": f"{probs[0].item():.4f}",
            }
            examples.append(example)

            # Log to logger
            logger.info(f"\n{'='*80}")
            logger.info(f"Example {len(examples)} (Epoch {epoch+1}):")
            logger.info(f"  Prompt: {example['prompt']}")
            logger.info(f"  Response A: {example['response_a']}")
            logger.info(f"  Response B: {example['response_b']}")
            logger.info(f"  True Label: {example['true_label']}")
            logger.info(f"  Predicted: {example['pred_label']} (confidence: {example['confidence']})")
            logger.info(f"  Prob(Same): {example['prob_same']} | Prob(Diff): {example['prob_diff']}")
            logger.info(f"  Correct: {example['correct']}")

    # Log to TensorBoard as text
    if writer is not None:
        for i, ex in enumerate(examples):
            text = f"""
**Example {i+1}**

**Prompt:** {ex['prompt']}

**Response A:** {ex['response_a']}

**Response B:** {ex['response_b']}

**True Label:** {ex['true_label']}
**Predicted:** {ex['pred_label']} ({ex['correct']})
**Confidence:** {ex['confidence']}
**P(Same):** {ex['prob_same']} | **P(Diff):** {ex['prob_diff']}
"""
            writer.add_text(f"examples/example_{i+1}", text, epoch)

    # Log to W&B as a table
    if cfg.use_wandb:
        table_data = [[
            ex["prompt"],
            ex["response_a"],
            ex["response_b"],
            ex["true_label"],
            ex["pred_label"],
            ex["confidence"],
            ex["correct"]
        ] for ex in examples]

        table = wandb.Table(
            columns=["Prompt", "Response A", "Response B", "True Label", "Predicted", "Confidence", "Correct"],
            data=table_data
        )
        wandb.log({f"examples/epoch_{epoch+1}": table, "epoch": epoch})

    model.train()


# -------------------- Evaluation -------------------- #

def evaluate_model(model, dataloader, device, epoch=None, writer=None, global_step=None):
    """Evaluate model on validation set."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")

    metrics = {
        "eval/loss": avg_loss,
        "eval/accuracy": accuracy,
        "eval/f1": f1,
    }

    # Log to tensorboard
    if writer is not None and global_step is not None:
        for key, value in metrics.items():
            writer.add_scalar(key, value, global_step)

    return metrics


# -------------------- Training -------------------- #

def train_epoch(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                device: str,
                epoch: int,
                cfg: Config,
                writer=None, global_step=0, scaler=None,
                on_epoch_end_callbacks:  List[Callable[[nn.Module, torch.utils.data.Dataset, Dict[str, Any]], None]] = []):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
    optimizer.zero_grad()

    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass with mixed precision
        if cfg.fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / cfg.accumulation_steps

            scaler.scale(loss).backward()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / cfg.accumulation_steps
            loss.backward()

        # Gradient accumulation
        if (step + 1) % cfg.accumulation_steps == 0:
            if cfg.fp16 and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        # Track metrics
        total_loss += loss.item() * cfg.accumulation_steps
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Logging
        if step % cfg.log_every_n_steps == 0:
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                "loss": loss.item() * cfg.accumulation_steps,
                "lr": f"{current_lr:.2e}"
            })

            if writer is not None:
                writer.add_scalar("train/loss", loss.item() * cfg.accumulation_steps, global_step)
                writer.add_scalar("train/learning_rate", current_lr, global_step)

            if cfg.use_wandb:
                wandb.log({
                    "train/loss": loss.item() * cfg.accumulation_steps,
                    "train/learning_rate": current_lr,
                    "train/step": global_step,
                })

    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")

    epoch_metrics = {
        "train/epoch_loss": avg_loss,
        "train/epoch_accuracy": accuracy,
        "train/epoch_f1": f1,
    }

    for fn in on_epoch_end_callbacks:
        fn(model, dataloader.dataset, epoch_metrics)


    return epoch_metrics, global_step


# -------------------- Main -------------------- #

def main():
    cfg = Config()
    set_seed(cfg.seed)
    ENTRIES_ARE_PAIRS = cfg.train_path != cfg.val_path

    # Create output directories
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.logging_dir, exist_ok=True)

    # Initialize logger
    log_file = os.path.join(cfg.output_dir, "training.log")
    logger = setup_logger("trainer", log_file=log_file)

    # Initialize tensorboard and wandb
    writer = None
    if cfg.use_tensorboard:
        writer = SummaryWriter(log_dir=cfg.logging_dir)

    if cfg.use_wandb:
        wandb.init(
            project=cfg.project_name,
            name=cfg.run_name,
            config=vars(cfg)
        )

    logger.info(f"Device: {cfg.device}")
    logger.info(f"Mixed precision training: {cfg.fp16}")

    # Load data
    logger.info("Loading data...")
    if ENTRIES_ARE_PAIRS:
        train_data, test_data = FunctionalPairDataset.train_test_split_jsonl(
            cfg.train_path,
            test_size=cfg.test_size,
            seed=cfg.seed
        )
    else:
        train_data = FunctionalPairDataset.load_jsonl(cfg.train_path)
        test_data = FunctionalPairDataset.load_jsonl(cfg.val_path)

    # Create datasets
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds = FunctionalPairDataset(
        train_data,
        tokenizer,
        num_pairs_per_iteration=cfg.num_pairs_per_epoch,
        max_length=cfg.max_length,
        neg_to_pos_ratio=cfg.neg_to_pos_ratio,
        entries_are_pairs=ENTRIES_ARE_PAIRS
    )
    test_ds = FunctionalPairDataset(
        test_data,
        tokenizer,
        num_pairs_per_iteration=cfg.num_pairs_per_epoch,
        max_length=cfg.max_length,
        neg_to_pos_ratio=None,
        entries_are_pairs=ENTRIES_ARE_PAIRS
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_bs,
        shuffle=True,
        num_workers=0,
        pin_memory=True if cfg.device == "cuda" else False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.eval_bs,
        shuffle=False,
        num_workers=0,
        pin_memory=True if cfg.device == "cuda" else False
    )

    logger.info(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # Load model
    logger.info(f"Loading model: {cfg.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2
    )

    # Apply LoRA if enabled
    if cfg.use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError(
                "LoRA training requires the 'peft' library. "
                "Install it with: pip install peft"
            )

        logger.info("Applying LoRA adapter...")
        logger.info(f"  LoRA rank (r): {cfg.lora_r}")
        logger.info(f"  LoRA alpha: {cfg.lora_alpha}")
        logger.info(f"  LoRA dropout: {cfg.lora_dropout}")
        logger.info(f"  Target modules: {cfg.lora_target_modules}")

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.to(cfg.device)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    total_steps = len(train_loader) * cfg.num_epochs // cfg.accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda") if cfg.fp16 else None

    # Training loop
    logger.info("\n" + "="*50)
    logger.info("Starting training...")
    logger.info("="*50 + "\n")

    best_f1 = 0.0
    global_step = 0

    for epoch in range(cfg.num_epochs):
        # Resample pairs for new epoch
        train_ds.on_epoch_end()

        # Train
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, cfg.device,
            epoch, cfg, writer, global_step, scaler, [lambda m, ds, mets: ds.on_epoch_end()]
        )

        logger.info(f"\nEpoch {epoch+1} Train Metrics:")
        for key, value in train_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        # Log train metrics
        if writer is not None:
            for key, value in train_metrics.items():
                writer.add_scalar(key, value, epoch)

        if cfg.use_wandb:
            wandb.log({**train_metrics, "epoch": epoch})

        # Log prediction examples
        logger.info(f"\nLogging prediction examples for epoch {epoch+1}...")
        log_prediction_examples(
            model, test_ds, tokenizer, cfg.device,
            epoch, cfg, logger, writer
        )

        # Evaluate
        if (epoch + 1) % cfg.eval_every_n_epochs == 0:
            eval_metrics = evaluate_model(
                model, test_loader, cfg.device,
                epoch, writer, global_step
            )

            logger.info(f"\nEpoch {epoch+1} Eval Metrics:")
            for key, value in eval_metrics.items():
                logger.info(f"  {key}: {value:.4f}")

            if cfg.use_wandb:
                wandb.log({**eval_metrics, "epoch": epoch})

            # Save best model
            if eval_metrics["eval/f1"] > best_f1:
                best_f1 = eval_metrics["eval/f1"]
                logger.info(f"\nNew best F1: {best_f1:.4f} - Saving model...")
                best_model_path = os.path.join(cfg.output_dir, "best_model")
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)

        # Save checkpoint
        if (epoch + 1) % cfg.save_every_n_epochs == 0:
            checkpoint_path = os.path.join(cfg.output_dir, f"checkpoint-epoch-{epoch+1}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Final save
    final_path = os.path.join(cfg.output_dir, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"\nTraining complete! Final model saved to {final_path}")
    logger.info(f"Best F1 score: {best_f1:.4f}")

    # Cleanup
    if writer is not None:
        writer.close()

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()