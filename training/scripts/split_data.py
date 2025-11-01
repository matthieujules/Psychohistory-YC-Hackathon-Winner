"""
Split curated cases into train/val sets (70/21 split)
"""

import json
from pathlib import Path
import random

# Load curated data
input_file = Path("training/data/curated_cases.jsonl")
train_file = Path("training/data/train.jsonl")
val_file = Path("training/data/val.jsonl")

cases = []
with open(input_file) as f:
    for line in f:
        cases.append(json.loads(line))

print(f"Loaded {len(cases)} cases")

# Shuffle with fixed seed for reproducibility
random.seed(42)
random.shuffle(cases)

# Split 70/21
split_idx = 70
train_cases = cases[:split_idx]
val_cases = cases[split_idx:]

# Write splits
with open(train_file, 'w') as f:
    for case in train_cases:
        f.write(json.dumps(case) + '\n')

with open(val_file, 'w') as f:
    for case in val_cases:
        f.write(json.dumps(case) + '\n')

print(f"✅ Split complete:")
print(f"   Train: {len(train_cases)} cases → {train_file}")
print(f"   Val:   {len(val_cases)} cases → {val_file}")
print(f"   Train examples: ~{len(train_cases) * 3}")
print(f"   Val examples: ~{len(val_cases) * 3}")
