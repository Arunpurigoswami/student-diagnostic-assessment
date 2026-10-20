from datasets import load_dataset
import pandas as pd

# Load SAF
ds = load_dataset(
    "Short-Answer-Feedback/saf_communication_networks_english"
)

train = ds["train"]

# Convert to DataFrame
df = train.to_pandas()

# Keep only non-perfect answers
candidates = df[df["score"] < 1.0].copy()

# Take up to 100 candidates
candidates = candidates.head(100)

# Save them
output_path = "data/saf_diagnostic_candidates.csv"
candidates.to_csv(output_path, index=False)

print(f"Saved {len(candidates)} candidate examples to {output_path}")
print("\nColumns:")
print(list(candidates.columns))

print("\nScore distribution:")
print(candidates["score"].value_counts().sort_index())