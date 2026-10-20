"""
Traditional Machine Learning Baseline:
TF-IDF + Logistic Regression for Student Answer Correctness Classification.

Experiments:
1. Baseline: provided_answer
2. Experiment A: question + provided_answer
3. Experiment B: question + reference_answer + provided_answer

IMPORTANT:
This is a shallow correctness classifier (Correct, Partially correct, Incorrect).
It DOES NOT predict reasoning error categories (conceptual, procedural, etc.).
"""

from pathlib import Path
from typing import Optional, Any
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import PROJECT_ROOT

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "correctness_classifier.joblib"
EXP_A_MODEL_PATH = DEFAULT_MODEL_DIR / "correctness_classifier_exp_a.joblib"
EXP_B_MODEL_PATH = DEFAULT_MODEL_DIR / "correctness_classifier_exp_b.joblib"

TARGET_CLASSES = ["Correct", "Partially correct", "Incorrect"]

CONTEXT_MODES = {
    "baseline": "Answer Only (Baseline)",
    "question_answer": "Question + Answer (Exp A)",
    "question_ref_answer": "Question + Reference + Answer (Exp B)"
}


def format_context_text(provided_answer: str,
                        question: str = "",
                        reference_answer: str = "",
                        mode: str = "baseline") -> str:
    """
    Format input text feature according to the experimental context mode.
    Modes:
      - 'baseline': provided_answer
      - 'question_answer': question + ' ' + provided_answer
      - 'question_ref_answer': question + ' ' + reference_answer + ' ' + provided_answer
    """
    ans = str(provided_answer).strip() if provided_answer else ""
    q = str(question).strip() if question else ""
    ref = str(reference_answer).strip() if reference_answer else ""

    if mode in ("question_answer", "exp_a"):
        return f"{q} {ans}".strip()
    elif mode in ("question_ref_answer", "exp_b"):
        return f"{q} {ref} {ans}".strip()
    else:  # baseline
        return ans


def build_classifier_pipeline(max_features: int = 5000,
                              ngram_range: tuple = (1, 2),
                              C: float = 1.0,
                              class_weight: str = "balanced",
                              random_state: int = 42) -> Pipeline:
    """
    Construct scikit-learn Pipeline with TfidfVectorizer and LogisticRegression.
    """
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                sublinear_tf=True
            )
        ),
        (
            "clf",
            LogisticRegression(
                C=C,
                max_iter=1000,
                class_weight=class_weight,
                random_state=random_state
            )
        )
    ])


def train_classifier(train_texts,
                     train_labels,
                     model_path: Path = DEFAULT_MODEL_PATH,
                     **pipeline_kwargs) -> Pipeline:
    """
    Fit the TF-IDF + LogisticRegression pipeline on training texts and labels,
    and persist the model artifact to disk.
    """
    pipeline = build_classifier_pipeline(**pipeline_kwargs)
    pipeline.fit(train_texts, train_labels)

    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    return pipeline


def load_classifier(model_path: Path = DEFAULT_MODEL_PATH) -> Optional[Pipeline]:
    """
    Load the fitted pipeline artifact from disk.
    Returns None if the artifact does not exist or fails to load.
    """
    path = Path(model_path)
    if not path.is_file():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def is_classifier_available(model_path: Path = DEFAULT_MODEL_PATH) -> bool:
    """Check if the fitted classifier artifact exists on disk."""
    return Path(model_path).is_file()


def predict_correctness(student_answer: str,
                        question: str = "",
                        reference_answer: str = "",
                        model: Optional[Any] = None,
                        model_path: Path = DEFAULT_MODEL_PATH,
                        mode: str = "baseline") -> dict:
    """
    Predict correctness category and class probability distribution.
    
    Returns a structured dictionary:
    - If model is missing: available=False, error message, no fake prediction.
    - If input is empty: available=True, error message, predicted_label=None.
    - If successful: available=True, predicted_label, probabilities, caveat.
    """
    # 1. Input Validation
    if not student_answer or not str(student_answer).strip():
        return {
            "available": is_classifier_available(model_path),
            "predicted_label": None,
            "probabilities": {},
            "model_type": f"TF-IDF + Logistic Regression ({CONTEXT_MODES.get(mode, mode)})",
            "error": "Student answer cannot be empty.",
            "caveat": "Statistical surface n-gram correlation only; predicts answer correctness tier, NOT reasoning error categories."
        }

    # 2. Check and Load Model
    clf = model
    if clf is None:
        clf = load_classifier(model_path)

    if clf is None:
        return {
            "available": False,
            "predicted_label": None,
            "probabilities": {},
            "model_type": f"TF-IDF + Logistic Regression ({CONTEXT_MODES.get(mode, mode)})",
            "error": f"ML classifier artifact not found at {model_path}. Please train the model using evaluation/train_classifier.py.",
            "caveat": "Statistical surface n-gram correlation only; predicts answer correctness tier, NOT reasoning error categories."
        }

    # 3. Format context text based on mode
    feature_text = format_context_text(
        provided_answer=student_answer,
        question=question,
        reference_answer=reference_answer,
        mode=mode
    )

    # 4. Predict Label and Class Probabilities
    predicted_label = clf.predict([feature_text])[0]

    prob_dict = {}
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba([feature_text])[0]
        classes = clf.classes_
        for cls_name, p in zip(classes, probs):
            prob_dict[str(cls_name)] = round(float(p), 4)

    return {
        "available": True,
        "predicted_label": str(predicted_label),
        "probabilities": prob_dict,
        "model_type": f"TF-IDF + Logistic Regression ({CONTEXT_MODES.get(mode, mode)})",
        "error": None,
        "caveat": "Statistical surface n-gram correlation only; predicts answer correctness tier, NOT reasoning error categories."
    }
