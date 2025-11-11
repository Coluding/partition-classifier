#!/usr/bin/env python3
import argparse
import json
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    fowlkes_mallows_score,
    precision_recall_fscore_support,
    accuracy_score,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import setup_logger

logger = setup_logger("evaluate_partitions")

# ----------------------------- Model I/O -----------------------------
def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    logger.info(f"Loading model from {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# ----------------------------- Data I/O -----------------------------
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(l) for l in f if l.strip()]

def _to_eval_item(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "generations" in raw and "partition" in raw:
        return {
            "id": raw.get("id", ""),
            "prompt": raw.get("prompt", ""),
            "generations": list(raw["generations"]),
            "partition": list(raw["partition"]),
            "is_pair": False,
            "label": None,
        }

    gens = []
    for idx, txt in sorted(
        ((int(k.split("_")[1]), v) for k, v in raw.items() if k.startswith("generation_")),
        key=lambda x: x[0]
    ):
        gens.append(txt)

    part = None
    if "similar" in raw and len(gens) == 2:
        part = [0, 0] if int(raw["similar"]) == 1 else [0, 1]

    return {
        "id": raw.get("id", ""),
        "prompt": raw.get("prompt", ""),
        "generations": gens,
        "partition": part if part is not None else [0 for _ in gens],
        "is_pair": True,
        "label": int(raw.get("similar", 0)) if len(gens) == 2 else None,
    }

def load_evaluation_data(data_path: str, num_samples: int | None = None, seed: int = 42):
    logger.info(f"Loading evaluation data from {data_path}...")
    raw = _read_jsonl(data_path)
    data = [_to_eval_item(r) for r in raw]

    all_pairs = all(d.get("is_pair", False) and len(d["generations"]) == 2 for d in data)

    if not all_pairs:
        data = [d for d in data if len(set(d["partition"])) >= 2]

    if num_samples and num_samples < len(data):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(data), num_samples, replace=False)
        data = [data[i] for i in idx]

    logger.info(f"Loaded {len(data)} items for evaluation")
    return data, all_pairs

# ----------------------------- Inference -----------------------------
@torch.no_grad()
def predict_pairwise_similarity(
    model,
    tokenizer,
    responses: List[str],
    device: str,
    max_length: int = 512,
    batch_size: int = 16,
    prompt: str = ""
) -> np.ndarray:
    n = len(responses)
    sim = np.eye(n, dtype=np.float32)
    sep = tokenizer.sep_token or "\n\n"

    pairs, idxs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            a = f"{prompt}{sep}{responses[i]}" if prompt else responses[i]
            b = f"{prompt}{sep}{responses[j]}" if prompt else responses[j]
            pairs.append((a, b))
            idxs.append((i, j))

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        a_list = [x[0] for x in batch]
        b_list = [x[1] for x in batch]
        enc = tokenizer(
            a_list, b_list,
            truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs_same = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        for (i, j), p in zip(idxs[start:start + batch_size], probs_same):
            sim[i, j] = sim[j, i] = float(p)

    return sim

# ----------------------------- Clustering -----------------------------
def _agglo_labels_precomputed(dist: np.ndarray, n_clusters: int, linkage: str) -> np.ndarray:
    try:
        # scikit-learn >= 1.4
        model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage=linkage)
    except TypeError:
        # scikit-learn <= 1.3
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage=linkage)
    return model.fit(dist).labels_

def cluster_from_similarity(
    sim: np.ndarray,
    use_true_k: bool,
    true_labels: np.ndarray,
    linkage: str = "average"
) -> np.ndarray:
    dist = 1.0 - sim

    if use_true_k:
        k = int(len(set(true_labels)))
    else:
        n = dist.shape[0]
        candidates = list(range(2, min(n, 10) + 1)) if n >= 2 else [1]
        best_k, best_score = candidates[0], -1e9
        for cand in candidates:
            pred = _agglo_labels_precomputed(dist, cand, linkage)
            avg_intra = np.mean([dist[i][pred == pred[i]].mean() for i in range(n)])
            score = -avg_intra
            if score > best_score:
                best_score, best_k = score, cand
        k = best_k

    return _agglo_labels_precomputed(dist, k, linkage)

# ----------------------------- Evaluation -----------------------------
def evaluate_single_prompt(
    model,
    tokenizer,
    item: Dict[str, Any],
    device: str,
    max_length: int = 512,
    batch_size: int = 16,
    use_true_k: bool = True,
    linkage: str = "average",
) -> Dict[str, float]:
    responses = item["generations"]
    true = np.array(item["partition"], dtype=int)

    sim = predict_pairwise_similarity(
        model, tokenizer, responses, device,
        max_length=max_length, batch_size=batch_size,
        prompt=item.get("prompt", "")
    )
    pred = cluster_from_similarity(sim, use_true_k, true, linkage)

    return {
        "ari": float(adjusted_rand_score(true, pred)),
        "nmi": float(normalized_mutual_info_score(true, pred)),
        "v": float(v_measure_score(true, pred)),
        "fmi": float(fowlkes_mallows_score(true, pred)),
    }

def evaluate_dataset(
    model,
    tokenizer,
    data: List[Dict[str, Any]],
    device: str,
    max_length: int = 512,
    batch_size: int = 16,
    use_true_k: bool = True,
    clustering_method: str = "average",
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    logger.info(f"Evaluating {len(data)} prompts...")
    per = [
        evaluate_single_prompt(
            model, tokenizer, item, device,
            max_length=max_length, batch_size=batch_size,
            use_true_k=use_true_k, linkage=clustering_method
        )
        for item in data
    ]

    def agg(key: str) -> Tuple[float, float, float]:
        vals = np.array([x[key] for x in per], dtype=np.float64)
        return float(vals.mean()), float(vals.std()), float(np.median(vals))

    summary = {
        "n_prompts": len(per),
        "ari": agg("ari"),
        "nmi": agg("nmi"),
        "v": agg("v"),
        "fmi": agg("fmi"),
    }
    return per, summary

@torch.no_grad()
def evaluate_pairs_dataset(model, tokenizer, data, device, max_length=800, batch_size=32):
    sep = tokenizer.sep_token or "\n\n"
    texts_a, texts_b, labels = [], [], []
    for it in data:
        prompt = it.get("prompt", "")
        a = it["generations"][0]
        b = it["generations"][1]
        ta = f"{prompt}{sep}{a}" if prompt else a
        tb = f"{prompt}{sep}{b}" if prompt else b
        texts_a.append(ta); texts_b.append(tb)
        labels.append(int(it["label"]))

    preds = []
    for s in range(0, len(texts_a), batch_size):
        enc = tokenizer(
            texts_a[s:s+batch_size],
            texts_b[s:s+batch_size],
            truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    summary = {"n_pairs": len(labels), "accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)}
    return summary

def print_summary(summary: Dict[str, Any]) -> None:
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Number of prompts evaluated: {summary['n_prompts']}")
    for name in ["ari", "nmi", "v", "fmi"]:
        mean, std, med = summary[name]
        label = {
            "ari": "Adjusted Rand Index (ARI)",
            "nmi": "Normalized Mutual Information (NMI)",
            "v": "V-measure",
            "fmi": "Fowlkes-Mallows Index (FMI)"
        }[name]
        logger.info(f"\n{label}:")
        logger.info(f"  Mean:   {mean:.4f} ± {std:.4f}")
        logger.info(f"  Median: {med:.4f}")

def save_results(results: List[Dict[str, float]], summary: Dict[str, Any], output_path: str) -> None:
    payload = {"summary": summary, "per_prompt": results}
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

# ----------------------------- CLI -----------------------------
def _parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate partition clustering with a pair classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    ap.add_argument("--data-path", type=str, required=True, help="JSONL with prompts/generations (or pairs)")
    ap.add_argument("--output-path", type=str, default="evaluation_results.json", help="Where to save results")
    ap.add_argument("--max-length", type=int, default=800, help="Max token length for tokenization")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    ap.add_argument("--no-use-true-k", action="store_true", help="Infer k instead of using ground-truth cluster count")
    ap.add_argument("--clustering-method", type=str, default="average", choices=["single", "complete", "average"])
    ap.add_argument("--num-samples", type=int, default=None, help="Evaluate only N prompts (random sample)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    ap.add_argument("--pairwise", action="store_true", help="Evaluate as pair classification (accuracy/F1)")
    return ap.parse_args()

def main():
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    data, all_pairs = load_evaluation_data(args.data_path, args.num_samples, args.seed)

    if args.pairwise or all_pairs:
        summary = evaluate_pairs_dataset(model, tokenizer, data, device, max_length=args.max_length, batch_size=args.batch_size)
        logger.info("\nPairwise Evaluation:")
        logger.info(f"  Pairs:     {summary['n_pairs']}")
        logger.info(f"  Accuracy:  {summary['accuracy']:.4f}")
        logger.info(f"  Precision: {summary['precision']:.4f}")
        logger.info(f"  Recall:    {summary['recall']:.4f}")
        logger.info(f"  F1:        {summary['f1']:.4f}")
        with open(args.output_path, "w") as f:
            json.dump({"pairwise_summary": summary}, f, indent=2)
        print("\nEvaluation complete!")
        return

    # Clustering path
    use_true_k = not args.no_use_true_k
    results, summary = evaluate_dataset(
        model, tokenizer, data, device,
        max_length=args.max_length, batch_size=args.batch_size,
        use_true_k=use_true_k, clustering_method=args.clustering_method
    )
    print_summary(summary)
    save_results(results, summary, args.output_path)
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()