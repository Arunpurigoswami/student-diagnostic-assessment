"""
Training and Evaluation Script for Traditional ML Baseline & Context Experiments:
TF-IDF + Logistic Regression on SAF Dataset for Answer Correctness Classification.

Configurations:
1. Baseline: provided_answer
2. Experiment A: question + provided_answer
3. Experiment B: question + reference_answer + provided_answer

Protocol:
- Train split: 1,700 samples (used for fitting candidate models)
- Validation split: 427 samples (used for candidate comparison and model selection)
- Test Unseen Answers: 375 samples (final evaluation of selected configuration)
- Test Unseen Questions: 479 samples (final evaluation of selected configuration)
"""

import sys
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_classifier import (
    train_classifier,
    format_context_text,
    DEFAULT_MODEL_PATH,
    EXP_A_MODEL_PATH,
    EXP_B_MODEL_PATH,
    TARGET_CLASSES,
)


def load_saf_splits():
    """Load official SAF communication networks dataset splits."""
    print("Loading SAF Communication Networks dataset...")
    ds = load_dataset("Short-Answer-Feedback/saf_communication_networks_english")
    return {
        "train": ds["train"].to_pandas(),
        "validation": ds["validation"].to_pandas(),
        "test_unseen_answers": ds["test_unseen_answers"].to_pandas(),
        "test_unseen_questions": ds["test_unseen_questions"].to_pandas(),
    }


def compute_majority_baseline(labels: pd.Series) -> float:
    """Calculate the accuracy achieved by always predicting the majority class."""
    if len(labels) == 0:
        return 0.0
    majority_count = labels.value_counts().iloc[0]
    return float(majority_count / len(labels))


def build_feature_series(df: pd.DataFrame, mode: str) -> pd.Series:
    """Vectorized feature construction for a given context mode."""
    if mode == "baseline":
        return df["provided_answer"].astype(str)
    elif mode in ("question_answer", "exp_a"):
        return df["question"].astype(str) + " " + df["provided_answer"].astype(str)
    elif mode in ("question_ref_answer", "exp_b"):
        return (
            df["question"].astype(str)
            + " "
            + df["reference_answer"].astype(str)
            + " "
            + df["provided_answer"].astype(str)
        )
    raise ValueError(f"Unknown mode: {mode}")


def evaluate_model(pipeline, X_series, y_series, split_name: str, config_name: str):
    """Compute complete classification evaluation metrics."""
    y_pred = pipeline.predict(X_series)

    acc = accuracy_score(y_series, y_pred)
    macro_f1 = f1_score(y_series, y_pred, average="macro")
    majority_acc = compute_majority_baseline(y_series)

    report_dict = classification_report(
        y_series,
        y_pred,
        labels=TARGET_CLASSES,
        output_dict=True,
        zero_division=0
    )
    conf_mat = confusion_matrix(y_series, y_pred, labels=TARGET_CLASSES)

    return {
        "config_name": config_name,
        "split_name": split_name,
        "n_samples": len(y_series),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "majority_baseline": majority_acc,
        "report_dict": report_dict,
        "confusion_matrix": conf_mat,
        "incorrect_precision": report_dict["Incorrect"]["precision"],
        "incorrect_recall": report_dict["Incorrect"]["recall"],
        "incorrect_f1": report_dict["Incorrect"]["f1-score"],
    }


def print_validation_comparison(results: list[dict]):
    """Format side-by-side comparison table of validation results."""
    print("\n" + "=" * 90)
    print("VALIDATION SET COMPARISON (Model Selection Protocol: N = 427)")
    print("=" * 90)
    print(f"{'Configuration':<32} | {'Accuracy':<9} | {'Macro-F1':<9} | {'Inc. Prec':<9} | {'Inc. Rec':<9} | {'Inc. F1':<8}")
    print("-" * 90)
    for res in results:
        cfg = res["config_name"]
        acc = f"{res['accuracy'] * 100:.2f}%"
        mf1 = f"{res['macro_f1']:.4f}"
        iprec = f"{res['incorrect_precision']:.3f}"
        irec = f"{res['incorrect_recall']:.3f}"
        if1 = f"{res['incorrect_f1']:.3f}"
        print(f"{cfg:<32} | {acc:<9} | {mf1:<9} | {iprec:<9} | {irec:<9} | {if1:<8}")
    print("-" * 90)
    print(f"Majority-Class Baseline: {results[0]['majority_baseline'] * 100:.2f}%\n")


def print_detailed_breakdown(res: dict):
    """Print class-by-class precision, recall, F1, and confusion matrix."""
    print(f"--- Detailed Breakdown: {res['config_name']} on {res['split_name']} ---")
    rep = res["report_dict"]
    for cls in TARGET_CLASSES:
        p = rep[cls]["precision"]
        r = rep[cls]["recall"]
        f = rep[cls]["f1-score"]
        s = int(rep[cls]["support"])
        print(f"  {cls:<18} -> Precision: {p:.3f}, Recall: {r:.3f}, F1: {f:.3f} (Support: {s})")
    print("  Confusion Matrix (Rows=True, Cols=Pred):")
    for i, row in enumerate(res["confusion_matrix"]):
        print(f"    {TARGET_CLASSES[i]:<18}: {row.tolist()}")
    print()


def main():
    print("=" * 90)
    print("TF-IDF + LOGISTIC REGRESSION: CONTEXT GENERALIZATION EXPERIMENTS")
    print("Task: Student Answer Correctness Classification")
    print("Labels: Correct | Partially correct | Incorrect")
    print("=" * 90)

    splits = load_saf_splits()

    # 1. Dataset Split Counts
    print("\n--- DATASET SPLIT & CLASS COUNTS ---")
    for s_name, s_df in splits.items():
        print(f"Split '{s_name}': {len(s_df)} samples")
        vc = s_df["verification_feedback"].value_counts()
        for cls in TARGET_CLASSES:
            cnt = vc.get(cls, 0)
            pct = (cnt / len(s_df)) * 100
            print(f"  - {cls:<18}: {cnt:>4} ({pct:>5.1f}%)")
        print()

    # Define experiment configurations
    configs = [
        {
            "name": "1. Baseline (provided_answer)",
            "mode": "baseline",
            "artifact": DEFAULT_MODEL_PATH
        },
        {
            "name": "2. Exp A (question + answer)",
            "mode": "question_answer",
            "artifact": EXP_A_MODEL_PATH
        },
        {
            "name": "3. Exp B (question + ref + ans)",
            "mode": "question_ref_answer",
            "artifact": EXP_B_MODEL_PATH
        }
    ]

    trained_pipelines = {}
    val_results = []

    # 2. Fit models strictly on train split
    train_df = splits["train"]
    val_df = splits["validation"]
    y_train = train_df["verification_feedback"]
    y_val = val_df["verification_feedback"]

    print("\n--- MODEL TRAINING & VALIDATION EVALUATION ---")
    for cfg in configs:
        mode = cfg["mode"]
        name = cfg["name"]
        art_path = cfg["artifact"]

        print(f"Fitting {name} on {len(train_df)} train samples...")
        X_train = build_feature_series(train_df, mode)
        pipeline = train_classifier(X_train, y_train, model_path=art_path)
        trained_pipelines[mode] = pipeline

        # Evaluate on validation
        X_val = build_feature_series(val_df, mode)
        res = evaluate_model(pipeline, X_val, y_val, "Validation", name)
        val_results.append(res)

    # 3. Print Validation Set Comparison
    print_validation_comparison(val_results)

    for res in val_results:
        print_detailed_breakdown(res)

    # 4. Model Selection based strictly on Validation Split
    # Find best config by Macro-F1 on validation
    best_val_res = max(val_results, key=lambda r: r["macro_f1"])
    print("=" * 90)
    print(f"MODEL SELECTION DECISION (Based strictly on Validation Split):")
    print(f"Selected Configuration: {best_val_res['config_name']}")
    print(f"Validation Macro-F1:    {best_val_res['macro_f1']:.4f}")
    print(f"Validation Accuracy:    {best_val_res['accuracy'] * 100:.2f}%")
    print("=" * 90)

    # 5. Evaluate on Official Held-Out Test Splits
    print("\n--- FINAL TEST SPLIT EVALUATION (Held-out, untouched during training) ---")
    test_splits = {
        "test_unseen_answers": splits["test_unseen_answers"],
        "test_unseen_questions": splits["test_unseen_questions"]
    }

    for t_name, t_df in test_splits.items():
        print(f"\n==================== TEST SPLIT: {t_name.upper()} (N = {len(t_df)}) ====================")
        y_test = t_df["verification_feedback"]

        # Evaluate all configurations for scientific completeness
        for cfg in configs:
            mode = cfg["mode"]
            name = cfg["name"]
            pipe = trained_pipelines[mode]
            X_test = build_feature_series(t_df, mode)
            t_res = evaluate_model(pipe, X_test, y_test, t_name, name)

            acc_str = f"{t_res['accuracy'] * 100:.2f}%"
            f1_str = f"{t_res['macro_f1']:.4f}"
            print(f"  {name:<32} -> Accuracy: {acc_str}, Macro-F1: {f1_str}, Inc F1: {t_res['incorrect_f1']:.3f}")
            print_detailed_breakdown(t_res)

    print("=" * 90)
    print("EXPERIMENTS COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    main()
