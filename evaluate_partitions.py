#!/usr/bin/env python3
"""
Evaluate partition prediction by clustering responses based on classifier predictions.

This script:
1. Loads a trained functional diversity classifier
2. Takes samples from the dataset (with ground truth partitions)
3. Predicts pairwise similarity between all responses for each prompt
4. Clusters responses into predicted partitions using the similarity matrix
5. Compares predicted partitions with ground truth using clustering metrics

Clustering metrics used:
- Adjusted Rand Index (ARI): measures similarity between clusterings
- Normalized Mutual Information (NMI): measures mutual dependence
- V-measure: harmonic mean of homogeneity and completeness
- Fowlkes-Mallows Index (FMI): geometric mean of pairwise precision and recall
"""

import argparse
import json
import os
import numpy as np
import torch
from typing import List, Dict, Tuple
from tqdm import tqdm
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    fowlkes_mallows_score,
)
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from utils import setup_logger


logger = setup_logger(__name__)


def load_model_and_tokenizer(model_path: str, device: str):
    """Load trained classifier and tokenizer."""
    logger.info(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def load_evaluation_data(data_path: str, num_samples: int = None, seed: int = 42):
    """Load JSONL data for evaluation."""
    logger.info(f"Loading evaluation data from {data_path}...")
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    if num_samples is not None and num_samples < len(data):
        np.random.seed(seed)
        indices = np.random.choice(len(data), num_samples, replace=False)
        data = [data[i] for i in indices]
        logger.info(f"Sampled {num_samples} items from dataset")

    logger.info(f"Loaded {len(data)} items for evaluation")
    return data


def predict_pairwise_similarity(
    model,
    tokenizer,
    responses: List[str],
    device: str,
    max_length: int = 512,
    batch_size: int = 32
) -> np.ndarray:
    """
    Predict pairwise similarity matrix for a set of responses.

    Returns:
        similarity_matrix: NxN matrix where element [i,j] is the probability
                          that responses i and j are in the same partition
    """
    n = len(responses)
    similarity_matrix = np.zeros((n, n))

    # Diagonal is always 1 (response is identical to itself)
    np.fill_diagonal(similarity_matrix, 1.0)

    # Generate all pairs
    pairs = []
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((responses[i], responses[j]))
            pair_indices.append((i, j))

    # Batch prediction
    all_probs = []
    with torch.no_grad():
        for batch_start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[batch_start:batch_start + batch_size]

            # Tokenize batch
            response_a = [p[0] for p in batch_pairs]
            response_b = [p[1] for p in batch_pairs]

            encodings = tokenizer(
                response_a,
                response_b,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )

            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            # Predict
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            # Extract probability of "same partition" (label=1)
            same_partition_probs = probs[:, 1].cpu().numpy()
            all_probs.extend(same_partition_probs)

    # Fill similarity matrix
    for (i, j), prob in zip(pair_indices, all_probs):
        similarity_matrix[i, j] = prob
        similarity_matrix[j, i] = prob  # Symmetric

    return similarity_matrix


def cluster_from_similarity(
    similarity_matrix: np.ndarray,
    num_clusters: int = None,
    method: str = "average"
) -> np.ndarray:
    """
    Cluster responses based on similarity matrix using hierarchical clustering.

    Args:
        similarity_matrix: NxN similarity matrix
        num_clusters: Number of clusters (if None, uses automatic threshold)
        method: Linkage method ('single', 'complete', 'average', 'ward')

    Returns:
        cluster_labels: array of cluster assignments
    """
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix

    # Ensure distance matrix is valid
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.clip(distance_matrix, 0, 1)

    # Convert to condensed form for scipy
    condensed_distances = squareform(distance_matrix, checks=False)

    # Hierarchical clustering
    linkage_matrix = linkage(condensed_distances, method=method)

    if num_clusters is None:
        # Use automatic threshold (e.g., at distance 0.5)
        cluster_labels = fcluster(linkage_matrix, t=0.5, criterion='distance')
    else:
        cluster_labels = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

    return cluster_labels


def evaluate_single_prompt(
    model,
    tokenizer,
    item: Dict,
    device: str,
    max_length: int = 512,
    batch_size: int = 32,
    use_true_k: bool = True,
    clustering_method: str = "average"
) -> Dict:
    """
    Evaluate partition prediction for a single prompt.

    Returns:
        metrics: dict with ARI, NMI, V-measure, FMI, and other info
    """
    responses = item["generations"]
    true_partitions = np.array(item["partition"])

    # Skip if only one response or all in same partition
    if len(responses) <= 1:
        return None

    # Predict similarity matrix
    similarity_matrix = predict_pairwise_similarity(
        model, tokenizer, responses, device, max_length, batch_size
    )

    # Determine number of clusters
    if use_true_k:
        num_clusters = len(np.unique(true_partitions))
    else:
        num_clusters = None

    # Cluster responses
    predicted_partitions = cluster_from_similarity(
        similarity_matrix,
        num_clusters=num_clusters,
        method=clustering_method
    )

    # Compute metrics
    ari = adjusted_rand_score(true_partitions, predicted_partitions)
    nmi = normalized_mutual_info_score(true_partitions, predicted_partitions)
    v_measure = v_measure_score(true_partitions, predicted_partitions)
    fmi = fowlkes_mallows_score(true_partitions, predicted_partitions)

    metrics = {
        "id": item["id"],
        "num_responses": len(responses),
        "num_true_partitions": len(np.unique(true_partitions)),
        "num_pred_partitions": len(np.unique(predicted_partitions)),
        "ari": ari,
        "nmi": nmi,
        "v_measure": v_measure,
        "fmi": fmi,
        "true_partitions": true_partitions.tolist(),
        "pred_partitions": predicted_partitions.tolist(),
        "similarity_matrix": similarity_matrix.tolist(),
    }

    return metrics


def evaluate_dataset(
    model,
    tokenizer,
    data: List[Dict],
    device: str,
    max_length: int = 512,
    batch_size: int = 32,
    use_true_k: bool = True,
    clustering_method: str = "average"
) -> Tuple[List[Dict], Dict]:
    """
    Evaluate partition prediction on entire dataset.

    Returns:
        results: list of per-prompt results
        summary: dict with aggregate metrics
    """
    results = []

    logger.info(f"Evaluating {len(data)} prompts...")
    for item in tqdm(data, desc="Evaluating prompts"):
        metrics = evaluate_single_prompt(
            model, tokenizer, item, device,
            max_length, batch_size, use_true_k, clustering_method
        )
        if metrics is not None:
            results.append(metrics)

    # Compute summary statistics
    ari_scores = [r["ari"] for r in results]
    nmi_scores = [r["nmi"] for r in results]
    v_measure_scores = [r["v_measure"] for r in results]
    fmi_scores = [r["fmi"] for r in results]

    summary = {
        "num_prompts_evaluated": len(results),
        "ari_mean": np.mean(ari_scores),
        "ari_std": np.std(ari_scores),
        "ari_median": np.median(ari_scores),
        "nmi_mean": np.mean(nmi_scores),
        "nmi_std": np.std(nmi_scores),
        "nmi_median": np.median(nmi_scores),
        "v_measure_mean": np.mean(v_measure_scores),
        "v_measure_std": np.std(v_measure_scores),
        "v_measure_median": np.median(v_measure_scores),
        "fmi_mean": np.mean(fmi_scores),
        "fmi_std": np.std(fmi_scores),
        "fmi_median": np.median(fmi_scores),
    }

    return results, summary


def save_results(results: List[Dict], summary: Dict, output_path: str):
    """Save evaluation results to file."""
    output = {
        "summary": summary,
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def print_summary(summary: Dict):
    """Print summary statistics."""
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Number of prompts evaluated: {summary['num_prompts_evaluated']}")
    logger.info("")
    logger.info(f"Adjusted Rand Index (ARI):")
    logger.info(f"  Mean:   {summary['ari_mean']:.4f} ± {summary['ari_std']:.4f}")
    logger.info(f"  Median: {summary['ari_median']:.4f}")
    logger.info("")
    logger.info(f"Normalized Mutual Information (NMI):")
    logger.info(f"  Mean:   {summary['nmi_mean']:.4f} ± {summary['nmi_std']:.4f}")
    logger.info(f"  Median: {summary['nmi_median']:.4f}")
    logger.info("")
    logger.info(f"V-measure:")
    logger.info(f"  Mean:   {summary['v_measure_mean']:.4f} ± {summary['v_measure_std']:.4f}")
    logger.info(f"  Median: {summary['v_measure_median']:.4f}")
    logger.info("")
    logger.info(f"Fowlkes-Mallows Index (FMI):")
    logger.info(f"  Mean:   {summary['fmi_mean']:.4f} ± {summary['fmi_std']:.4f}")
    logger.info(f"  Median: {summary['fmi_median']:.4f}")
    logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate partition prediction using trained classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to JSONL data file with ground truth partitions"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for prediction"
    )
    parser.add_argument(
        "--clustering-method",
        type=str,
        default="average",
        choices=["single", "complete", "average", "ward"],
        help="Hierarchical clustering linkage method"
    )
    parser.add_argument(
        "--no-use-true-k",
        action="store_true",
        help="Don't use ground truth number of clusters (use automatic detection)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on"
    )

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    # Load evaluation data
    data = load_evaluation_data(args.data_path, args.num_samples, args.seed)

    # Evaluate
    results, summary = evaluate_dataset(
        model,
        tokenizer,
        data,
        args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_true_k=not args.no_use_true_k,
        clustering_method=args.clustering_method
    )

    # Print and save results
    print_summary(summary)
    save_results(results, summary, args.output_path)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()