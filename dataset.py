import json
import random
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer
)

class FunctionalPairDataset(Dataset):
    """
    Dynamically generates balanced positive/negative response pairs from a JSONL file.
    Supports safe prompt-level train/test splitting.
    """

    def __init__(self, jsonl_path_or_data, tokenizer, n_pairs_per_prompt=10, max_length=512,
                 num_pairs_per_iteration: int = 100, balance=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_pairs = n_pairs_per_prompt
        self.balance = balance
        self.num_pairs_per_iteration = num_pairs_per_iteration

        # Accept list of dicts OR path to JSONL
        if isinstance(jsonl_path_or_data, str):
            self.data = self._load_jsonl(jsonl_path_or_data)
        else:
            self.data = jsonl_path_or_data  # already loaded list of dicts

        self._pairs_to_sample = None

        self._prepare_all_partitions()
        self._resample_pairs()

    # ---------- Static Utilities ---------- #

    @staticmethod
    def _load_jsonl(path):
        data = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    @staticmethod
    def train_test_split_jsonl(jsonl_path, test_size=0.2, seed=42):
        """
        Splits the JSONL file at the prompt level (no data leakage).
        Returns (train_data, test_data) as Python lists.
        """
        data = FunctionalPairDataset._load_jsonl(jsonl_path)
        random.Random(seed).shuffle(data)
        split_idx = int(len(data) * (1 - test_size))
        train_data, test_data = data[:split_idx], data[split_idx:]
        print(f"Split: {len(train_data)} train prompts, {len(test_data)} test prompts")
        return train_data, test_data

    # ---------- Core Dataset Methods ---------- #

    def _prepare_all_partitions(self):
        """Precompute partition → indices per prompt."""
        for item in self.data:
            partition_map = {}
            for idx, part in enumerate(item["partition"]):
                partition_map.setdefault(part, []).append(idx)
            item["_partition_map"] = partition_map

    def _resample_pairs(self):
        """Resample pairs for the next epoch."""
        pairs = []
        for item in self.data:
            gens = item["generations"]
            parts = item["_partition_map"]

            positives, negatives = [], []

            # --- Positive pairs (same partition)
            for part, idxs in parts.items():
                if len(idxs) > 1:
                    for i in range(len(idxs)):
                        for j in range(i + 1, len(idxs)):
                            positives.append((idxs[i], idxs[j]))

            # --- Negative pairs (different partitions)
            all_ids = list(range(len(gens)))
            for i in range(len(all_ids)):
                for j in range(i + 1, len(all_ids)):
                    if item["partition"][i] != item["partition"][j]:
                        negatives.append((i, j))

            # --- Balanced sampling
            if self.balance:
                n_half = self.n_pairs // 2
                pos_sample = random.sample(positives, min(len(positives), n_half)) if positives else []
                neg_sample = random.sample(negatives, min(len(negatives), n_half))
                sampled = [(i, j, 1) for i, j in pos_sample] + [(i, j, 0) for i, j in neg_sample]
            else:
                all_pairs = [(i, j, 1 if item["partition"][i] == item["partition"][j] else 0)
                             for i in range(len(gens)) for j in range(i + 1, len(gens))]
                sampled = random.sample(all_pairs, min(len(all_pairs), self.n_pairs))

            for i, j, label in sampled:
                pairs.append({
                    "response_a": gens[i],
                    "response_b": gens[j],
                    "label": label,
                    "prompt": item["prompt"],
                    "id": item["id"]
                })

        random.shuffle(pairs)
        self.pairs = pairs
        self._pairs_to_sample = self.pairs[:self.num_pairs_per_iteration]

    def on_epoch_end(self):
        """Recreate new pairs for a new epoch."""
        self._resample_pairs()

    def __len__(self):
        return len(self._pairs_to_sample)

    def __getitem__(self, idx):
        item = self._pairs_to_sample[idx]
        enc = self.tokenizer(
            item["response_a"],
            item["response_b"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(item["label"], dtype=torch.long)
        return enc


if __name__ == "__main__":

    ds = FunctionalPairDataset("data/scores.jsonl",
                               tokenizer=AutoTokenizer.from_pretrained("microsoft/deberta-v3-large"),)

    item = ds[10]