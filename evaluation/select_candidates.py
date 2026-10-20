import pandas as pd

df = pd.read_csv("data/saf_diagnostic_candidates.csv")

# Select examples from different score levels
scores = [0.0, 0.25, 0.5, 0.75]

selected = []

for score in scores:
    group = df[df["score"] == score].head(5)
    selected.append(group)

result = pd.concat(selected, ignore_index=True)

# Preserve all diagnostic evaluation fields including reference_answer and verification_feedback
columns_to_keep = [
    "id",
    "question",
    "reference_answer",
    "provided_answer",
    "answer_feedback",
    "verification_feedback",
    "score",
]

result = result[columns_to_keep]

result.to_csv("data/diagnostic_shortlist.csv", index=False)

print(f"Created shortlist with {len(result)} examples.")
print("Columns preserved:", list(result.columns))
print("Saved to: data/diagnostic_shortlist.csv")
