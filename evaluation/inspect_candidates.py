import pandas as pd

df = pd.read_csv("data/saf_diagnostic_candidates.csv")

pd.set_option("display.max_colwidth", 120)

for i, row in df.iterrows():
    print("\n" + "=" * 100)
    print(f"CANDIDATE {i + 1}")
    print(f"ID: {row['id']}")
    print(f"SCORE: {row['score']}")

    print("\nQUESTION:")
    print(row["question"])

    print("\nSTUDENT ANSWER:")
    print(row["provided_answer"])

    print("\nFEEDBACK:")
    print(row["answer_feedback"])