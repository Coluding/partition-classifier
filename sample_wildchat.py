from datasets import load_dataset

OUT_DIR = "/home/scur1900/partition-classifier/datasets"
N_TRAIN = 10000
N_VAL = 2000

print("Loading allenai/WildChat-1M...")
ds = load_dataset("allenai/WildChat-1M")
train = ds["train"]

def extract_prompt(example):
    conv = example.get("conversation", None)
    if conv and len(conv) > 0 and "content" in conv[0]:
        return {"prompt": conv[0]["content"]}
    return {"prompt": None}

print("Extracting prompts...")
train = train.map(extract_prompt)

train = train.filter(lambda x: x["prompt"] is not None)

# Shuffle full dataset once
train = train.shuffle(seed=42)

# Split
train_split = train.select(range(N_TRAIN))
val_split = train.select(range(N_TRAIN, N_TRAIN + N_VAL))

train_out = f"{OUT_DIR}/wildchat10k.parquet"
val_out = f"{OUT_DIR}/wildchat_valid.parquet"

train_split.to_parquet(train_out)
val_split.to_parquet(val_out)

print("Saved train to:", train_out)
print("Saved val   to:", val_out)
