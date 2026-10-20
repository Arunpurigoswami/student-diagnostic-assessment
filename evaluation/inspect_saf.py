from datasets import load_dataset
from collections import Counter

# Load SAF
ds = load_dataset(
    "Short-Answer-Feedback/saf_communication_networks_english"
)

train = ds["train"]

# Score distribution
scores = Counter(train["score"])

print("\n=== SCORE DISTRIBUTION ===")
for score, count in sorted(scores.items()):
    print(f"Score {score}: {count} examples")

# Show examples from important score groups
target_scores = [0.0, 0.5, 0.75, 1.0]

print("\n=== SAMPLE DIAGNOSTIC EXAMPLES ===")

for target_score in target_scores:
    print(f"\n\n========== SCORE {target_score} ==========")

    count = 0

    for example in train:
        if example["score"] == target_score:
            print("\nQuestion:")
            print(example["question"])

            print("\nStudent answer:")
            print(example["provided_answer"])

            print("\nReference answer:")
            print(example["reference_answer"])

            print("\nFeedback:")
            print(example["answer_feedback"])

            count += 1

            if count == 3:
                break