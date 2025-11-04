#!/usr/bin/env python3
"""
Main entry point for training the functional diversity classifier.
Supports command-line argument parsing to override default configuration.
"""

import argparse
import sys
from train import Config, main as train_main
import train


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a functional diversity classifier for LLM response pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data parameters
    parser.add_argument(
        "--data-path",
        type=str,
        default=Config.data_path,
        help="Path to JSONL data file"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=Config.test_size,
        help="Fraction of data to use for testing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=Config.seed,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num-pairs-per-epoch",
        type=int,
        default=Config.num_pairs_per_epoch,
        help="Number of pairs to sample per epoch"
    )
    parser.add_argument(
        "--n-pairs-per-prompt",
        type=int,
        default=Config.n_pairs_per_prompt,
        help="Number of pairs to sample per prompt"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=Config.max_length,
        help="Maximum sequence length for tokenization"
    )

    # Model parameters
    parser.add_argument(
        "--model-name",
        type=str,
        default=Config.model_name,
        help="HuggingFace model name or path"
    )

    # Training parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=Config.lr,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=Config.num_epochs,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--train-bs",
        type=int,
        default=Config.train_bs,
        help="Training batch size"
    )
    parser.add_argument(
        "--eval-bs",
        type=int,
        default=Config.eval_bs,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=Config.weight_decay,
        help="Weight decay for AdamW optimizer"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=Config.warmup_steps,
        help="Number of warmup steps for learning rate scheduler"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=Config.max_grad_norm,
        help="Maximum gradient norm for clipping"
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=Config.accumulation_steps,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable mixed precision training (FP16)"
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default=Config.output_dir,
        help="Directory to save model checkpoints and logs"
    )
    parser.add_argument(
        "--logging-dir",
        type=str,
        default=Config.logging_dir,
        help="Directory for TensorBoard logs"
    )

    # Logging parameters
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging"
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=Config.project_name,
        help="W&B project name"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=Config.run_name,
        help="W&B run name"
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=Config.log_every_n_steps,
        help="Log training metrics every N steps"
    )
    parser.add_argument(
        "--eval-every-n-epochs",
        type=int,
        default=Config.eval_every_n_epochs,
        help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=Config.save_every_n_epochs,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--num-examples-to-log",
        type=int,
        default=Config.num_examples_to_log,
        help="Number of prediction examples to log each epoch"
    )

    return parser.parse_args()


def update_config_from_args(cfg, args):
    """Update Config object with command-line arguments."""
    # Data parameters
    cfg.data_path = args.data_path
    cfg.test_size = args.test_size
    cfg.seed = args.seed
    cfg.num_pairs_per_epoch = args.num_pairs_per_epoch
    cfg.n_pairs_per_prompt = args.n_pairs_per_prompt
    cfg.max_length = args.max_length

    # Model parameters
    cfg.model_name = args.model_name

    # Training parameters
    cfg.lr = args.lr
    cfg.num_epochs = args.num_epochs
    cfg.train_bs = args.train_bs
    cfg.eval_bs = args.eval_bs
    cfg.weight_decay = args.weight_decay
    cfg.warmup_steps = args.warmup_steps
    cfg.max_grad_norm = args.max_grad_norm
    cfg.accumulation_steps = args.accumulation_steps

    # FP16 flag (inverted logic)
    if args.no_fp16:
        cfg.fp16 = False

    # Output parameters
    cfg.output_dir = args.output_dir
    cfg.logging_dir = args.logging_dir

    # Logging parameters
    cfg.use_wandb = args.use_wandb
    cfg.use_tensorboard = not args.no_tensorboard
    cfg.project_name = args.project_name
    cfg.run_name = args.run_name
    cfg.log_every_n_steps = args.log_every_n_steps
    cfg.eval_every_n_epochs = args.eval_every_n_epochs
    cfg.save_every_n_epochs = args.save_every_n_epochs
    cfg.num_examples_to_log = args.num_examples_to_log

    return cfg


def main():
    """Main entry point with argument parsing."""
    args = parse_args()

    # Create and update config
    cfg = Config()
    cfg = update_config_from_args(cfg, args)
    train.Config = type('Config', (), vars(cfg))

    # Run training
    train_main()


if __name__ == "__main__":
    main()