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

    def __init__(self, jsonl_path_or_data, tokenizer, max_length=512,
                 num_pairs_per_iteration: int = 100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_pairs_per_iteration = num_pairs_per_iteration

        # Accept list of dicts OR path to JSONL
        if isinstance(jsonl_path_or_data, str):
            self.data = self._load_jsonl(jsonl_path_or_data)
        else:
            self.data = jsonl_path_or_data  # already loaded list of dicts


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
        """Rejection sampling: build a pool of pairs from different partitions only, globally."""
        pairs = []
        for item in self.data:
            gens = item["generations"]
            parts = item["_partition_map"]
            for p1, p2 in combinations(parts.keys(), 2):
                for i in parts[p1]:
                    for j in parts[p2]:
                        pairs.append({
                            "response_a": gens[i],
                            "response_b": gens[j],
                            "label": 0,  # different partitions
                            "prompt": item["prompt"],
                            "id": item["id"]
                        })
        random.shuffle(pairs)
        self.pairs = pairs[:min(self.num_pairs_per_iteration, len(pairs))]
        self._pairs_to_sample = self.pairs

    def __iter__(self):
        while True:
            for item in self._pairs_to_sample:
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
                yield enc
            # After one epoch, rebuild the pool (rejection sampling)
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
    ds = FunctionalPairDataset(
        "data/scores.jsonl",
        tokenizer=AutoTokenizer.from_pretrained("microsoft/deberta-v3-large"),
        num_pairs_per_iteration=100
    )
    for i, item in enumerate(ds):
        print(item)
        if i >= 99:  # num_pairs_per_iteration - 1
            break