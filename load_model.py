#!/usr/bin/env python3
"""
Utility script to load trained models (both full fine-tuned and LoRA).
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


from peft import PeftModel
PEFT_AVAILABLE = True


def load_model(model_path, is_lora=False, base_model=None, device="cuda"):
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint directory
        is_lora: Whether this is a LoRA checkpoint
        base_model: Base model name (required if is_lora=True)
        device: Device to load model on

    Returns:
        model, tokenizer
    """
    if is_lora:
        if not PEFT_AVAILABLE:
            raise ImportError(
                "Loading LoRA models requires 'peft' library. "
                "Install with: pip install peft"
            )

        if base_model is None:
            raise ValueError(
                "base_model must be specified when loading LoRA checkpoints. "
                "Example: --base-model microsoft/deberta-v3-large"
            )

        print(f"Loading base model: {base_model}")
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=2
        )

        print(f"Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)

        # Merge LoRA weights into base model (optional, for deployment)
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        print(f"Loading full model from: {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    return model, tokenizer


def test_inference(model, tokenizer, device="cuda"):
    """Test inference with example response pair."""
    response_a = "To solve this problem, we can use dynamic programming."
    response_b = "A dynamic programming approach would work well here."

    # Tokenize
    inputs = tokenizer(
        response_a,
        response_b,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_label = torch.argmax(logits, dim=-1).item()

    print("\n" + "="*60)
    print("Test Inference:")
    print("="*60)
    print(f"Response A: {response_a}")
    print(f"Response B: {response_b}")
    print(f"\nPrediction: {'Same Partition' if pred_label == 1 else 'Different Partition'}")
    print(f"Confidence: {probs[pred_label].item():.4f}")
    print(f"P(Same): {probs[1].item():.4f} | P(Diff): {probs[0].item():.4f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Load and test a trained functional diversity classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Load as LoRA checkpoint (requires --base-model)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (required for LoRA checkpoints)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load model on"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test inference after loading"
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(
        model_path=args.model_path,
        is_lora=args.lora,
        base_model=args.base_model,
        device=args.device
    )

    # Test inference
    if args.test:
        test_inference(model, tokenizer, args.device)

    print("\nModel ready for inference!")
    return model, tokenizer


if __name__ == "__main__":
    main()