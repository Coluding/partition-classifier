import json
import random
import torch
from typing import Optional
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils import setup_logger

logger = setup_logger(__name__)

class FunctionalPairDataset(Dataset):
    """
    Builds a global pool of ALL possible response pairs across all prompts.
    At the start of each epoch, samples a subset using rejection sampling
    to achieve a specified positive/negative ratio.
    """

    def __init__(self, jsonl_path_or_data, tokenizer, max_length=512,
                 num_pairs_per_iteration: int = 1000, neg_to_pos_ratio: Optional[float] = 0.5,
                 entries_are_pairs: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_pairs_per_iteration = num_pairs_per_iteration
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.entries_are_pairs = entries_are_pairs

        # Load dataset (JSONL or list)
        if isinstance(jsonl_path_or_data, str):
            self.data = self.load_jsonl(jsonl_path_or_data)
        else:
            self.data = jsonl_path_or_data

        if self.entries_are_pairs:
            self._prepare_pairs()
        else:
            self._prepare_all_partitions()
            self._build_global_pool()
        self._resample_pairs()

    @staticmethod
    def load_jsonl(path):
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def train_test_split_jsonl(jsonl_path, test_size=0.2, seed=42):
        data = FunctionalPairDataset.load_jsonl(jsonl_path)
        random.Random(seed).shuffle(data)
        split_idx = int(len(data) * (1 - test_size))
        return data[:split_idx], data[split_idx:]

    # ---------- Core methods ---------- #

    def _prepare_pairs(self):
        logger.info("Preparing pairs from provided entries...")
        all_pairs = []
        for item in self.data:
            all_pairs.append({
                "response_a": item["generation_0"],
                "response_b": item["generation_1"],
                "label": item["similar"],
                "prompt": item.get("prompt", ""),
                "id": item.get("id", "")
            })
        self._all_pairs = all_pairs
        logger.info(f"Total pairs from entries: {len(all_pairs)}")

    def _prepare_all_partitions(self):
        """Precompute partition → indices for each prompt."""
        for item in self.data:
            partition_map = {}
            for idx, part in enumerate(item["partition"]):
                partition_map.setdefault(part, []).append(idx)
            item["_partition_map"] = partition_map

    def _build_global_pool(self):
        """
        Build a global pool of ALL unique pairs across all prompts.
        Labels:
          - 1 = same partition
          - 0 = different partitions
        """
        logger.info("Building global pair pool...")
        all_pairs = []

        for item in self.data:
            gens = item["generations"]
            parts = item["_partition_map"]
            n = len(gens)


            for i in range(n):
                for j in range(i + 1, n):
                    label = int(item["partition"][i] == item["partition"][j])
                    all_pairs.append({
                        "response_a": gens[i],
                        "response_b": gens[j],
                        "label": label,
                        "prompt": item["prompt"],
                        "id": item["id"]
                    })

        self._all_pairs = all_pairs
        logger.info(f"Global pool size: {len(all_pairs)} total pairs")

    def _resample_pairs(self):
        """
        Rejection sample pairs for one epoch based on desired neg/pos ratio.
        e.g., 0.75 => 75% negatives, 25% positives.
        """

        if self.neg_to_pos_ratio is None:
            self._pairs_to_sample = self._all_pairs
            return

        positives = [p for p in self._all_pairs if p["label"] == 1]
        negatives = [p for p in self._all_pairs if p["label"] == 0]

        neg_pos_ratio = random.uniform(self.neg_to_pos_ratio - 0.1, self.neg_to_pos_ratio + 0.1)

        num_neg = int(self.num_pairs_per_iteration * neg_pos_ratio)
        num_pos = self.num_pairs_per_iteration - num_neg

        neg_sample = random.sample(negatives, min(len(negatives), num_neg))
        pos_sample = random.sample(positives, min(len(positives), num_pos))

        self._pairs_to_sample = pos_sample + neg_sample
        random.shuffle(self._pairs_to_sample)

        logger.info(f"Sampled {len(pos_sample)} positives, {len(neg_sample)} negatives "
              f"({len(self._pairs_to_sample)} total for this epoch)")

    def on_epoch_end(self):
        """Calling this after an epoch has finished to get fresh pairs"""
        self._resample_pairs()

    def __len__(self):
        return len(self._pairs_to_sample)

    def __getitem__(self, idx):
        item = self._pairs_to_sample[idx]

        # Prepend prompt as context
        prompt = (item.get("prompt") or "").strip()
        sep = self.tokenizer.sep_token or "\n\n"

        # text_a =  item["response_a"]
        # text_b =  item["response_b"]

        text_a = f"{prompt}{sep}{item['response_a']}" if prompt else item["response_a"]
        text_b = f"{prompt}{sep}{item['response_b']}" if prompt else item["response_b"]

        enc = self.tokenizer(
            text_a,
            text_b,
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
        num_pairs_per_iteration=5000
    )
    print(ds[10])
    ds.on_epoch_end()
    print(ds[10])