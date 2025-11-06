#!/usr/bin/env python3
"""
Interactive dashboard for evaluating the functional diversity classifier.
Allows testing response pairs and viewing predictions in real-time.
"""

import argparse
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional: LoRA support
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class FunctionalDiversityPredictor:
    """Wrapper class for model inference."""

    def __init__(self, model_path, is_lora=False, base_model=None, device="cuda"):
        """Initialize the predictor with a trained model."""
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.is_lora = is_lora

        print(f"Loading model on {self.device}...")

        if is_lora:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "Loading LoRA models requires 'peft' library. "
                    "Install with: pip install peft"
                )

            if base_model is None:
                raise ValueError("base_model must be specified for LoRA checkpoints")

            print(f"  Base model: {base_model}")
            print(f"  LoRA adapter: {model_path}")

            model = AutoModelForSequenceClassification.from_pretrained(
                base_model,
                num_labels=2
            )
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            print(f"  Model path: {model_path}")
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!\n")

    def predict(self, response_a, response_b, max_length=1024):
        """
        Predict whether two responses belong to the same functional partition.

        Args:
            response_a: First response text
            response_b: Second response text
            max_length: Maximum sequence length for tokenization

        Returns:
            Dictionary with prediction results
        """
        # Validate inputs
        if not response_a or not response_a.strip():
            return {
                "error": "Response A is empty",
                "prediction": None,
                "confidence": None,
                "prob_same": None,
                "prob_different": None,
            }

        if not response_b or not response_b.strip():
            return {
                "error": "Response B is empty",
                "prediction": None,
                "confidence": None,
                "prob_same": None,
                "prob_different": None,
            }

        # Tokenize
        inputs = self.tokenizer(
            response_a,
            response_b,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_label = torch.argmax(logits, dim=-1).item()

        # Format results
        prediction = "Same Partition" if pred_label == 1 else "Different Partition"
        confidence = probs[pred_label].item()
        prob_same = probs[1].item()
        prob_different = probs[0].item()

        return {
            "error": None,
            "prediction": prediction,
            "confidence": confidence,
            "prob_same": prob_same,
            "prob_different": prob_different,
        }


def create_dashboard(predictor):
    """Create Gradio interface for the dashboard."""

    def predict_fn(response_a, response_b):
        """Prediction function for Gradio interface."""
        result = predictor.predict(response_a, response_b)

        if result["error"]:
            return f"Error: {result['error']}", "", "", ""

        # Format prediction with emoji
        pred_emoji = "✅" if result["prediction"] == "Same Partition" else "❌"
        prediction_text = f"{pred_emoji} **{result['prediction']}**"

        # Format confidence as percentage
        confidence_text = f"{result['confidence']*100:.2f}%"

        # Format probabilities
        prob_same_text = f"{result['prob_same']*100:.2f}%"
        prob_diff_text = f"{result['prob_different']*100:.2f}%"

        return prediction_text, confidence_text, prob_same_text, prob_diff_text

    # Example pairs
    examples = [
        [
            "To solve this problem, we can use dynamic programming with memoization.",
            "A dynamic programming approach with caching would be efficient here."
        ],
        [
            "The answer is 42.",
            "To solve this, first we need to understand the problem constraints and then apply a greedy algorithm."
        ],
        [
            "You should use a hash map to store the values for O(1) lookup.",
            "A dictionary (hash table) would give you constant-time access to the values."
        ],
        [
            "The time complexity is O(n log n) due to the sorting step.",
            "This runs in O(n^2) time because of the nested loops."
        ],
    ]

    # Create Gradio interface
    with gr.Blocks(title="Functional Diversity Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Functional Diversity Classifier

            Classify whether two LLM-generated responses belong to the same **functional partition**.

            while responses in different partitions use different approaches (e.g., DP vs greedy).

            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                response_a = gr.Textbox(
                    label="Response A",
                    placeholder="Enter the first response here...",
                    lines=6,
                    max_lines=10,
                )
                response_b = gr.Textbox(
                    label="Response B",
                    placeholder="Enter the second response here...",
                    lines=6,
                    max_lines=10,
                )

                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    predict_btn = gr.Button("Predict", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Prediction Results")

                prediction_output = gr.Markdown(
                    label="Prediction",
                    value="*Enter two responses and click Predict*"
                )

                with gr.Row():
                    confidence_output = gr.Textbox(
                        label="Confidence",
                        interactive=False,
                        scale=1
                    )

                gr.Markdown("### 📈 Probabilities")

                with gr.Row():
                    prob_same_output = gr.Textbox(
                        label="P(Same Partition)",
                        interactive=False,
                        scale=1
                    )
                    prob_diff_output = gr.Textbox(
                        label="P(Different Partition)",
                        interactive=False,
                        scale=1
                    )

        gr.Markdown("### 💡 Example Pairs")
        gr.Examples(
            examples=examples,
            inputs=[response_a, response_b],
            label="Click an example to load it"
        )

        # Event handlers
        predict_btn.click(
            fn=predict_fn,
            inputs=[response_a, response_b],
            outputs=[prediction_output, confidence_output, prob_same_output, prob_diff_output]
        )

        clear_btn.click(
            fn=lambda: ("", "", "*Enter two responses and click Predict*", "", "", ""),
            inputs=[],
            outputs=[response_a, response_b, prediction_output, confidence_output, prob_same_output, prob_diff_output]
        )

        gr.Markdown(
            """
            ---
            **Model Info:**
            - Model Path: `{}`
            - Type: {}
            - Device: {}
            """.format(
                predictor.model_path,
                "LoRA Fine-tuned" if predictor.is_lora else "Full Fine-tuned",
                predictor.device
            )
        )

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Launch interactive dashboard for functional diversity classifier",
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
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on"
    )

    args = parser.parse_args()

    # Initialize predictor
    predictor = FunctionalDiversityPredictor(
        model_path=args.model_path,
        is_lora=args.lora,
        base_model=args.base_model,
        device=args.device
    )

    # Create and launch dashboard
    demo = create_dashboard(predictor)

    print("\n" + "="*60)
    print("Launching dashboard...")
    print("="*60)

    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()