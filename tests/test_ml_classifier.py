"""
Unit tests for Traditional ML Baseline & Context Experiments (TF-IDF + Logistic Regression):
Tests training, model saving/loading, predictions, probability outputs,
context formatting, invalid input handling, and missing model artifact handling.
"""

import unittest
from pathlib import Path
import tempfile
import shutil

from src.ml_classifier import (
    build_classifier_pipeline,
    train_classifier,
    load_classifier,
    is_classifier_available,
    predict_correctness,
    format_context_text,
    TARGET_CLASSES,
)


class TestMLClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for test model artifacts
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.test_model_path = cls.test_dir / "test_model.joblib"
        cls.test_exp_a_path = cls.test_dir / "test_model_exp_a.joblib"

        # Synthetic small training dataset
        cls.train_answers = [
            "This is a completely correct answer with full explanation.",
            "Accurate and valid definition of routing protocol.",
            "Partially right but missing the key definition.",
            "Somewhat correct, although explanation is incomplete.",
            "Totally wrong answer with incorrect statements.",
            "False and invalid explanation.",
        ]
        cls.train_questions = [
            "What is routing?",
            "Explain routing protocol.",
            "Define latency.",
            "Describe throughput.",
            "How does switch work?",
            "What is hub?",
        ]
        cls.train_labels = [
            "Correct",
            "Correct",
            "Partially correct",
            "Partially correct",
            "Incorrect",
            "Incorrect",
        ]

        # Train baseline test model
        cls.test_pipeline = train_classifier(
            cls.train_answers,
            cls.train_labels,
            model_path=cls.test_model_path,
            max_features=100
        )

        # Train Exp A test model (question + answer)
        exp_a_texts = [f"{q} {a}" for q, a in zip(cls.train_questions, cls.train_answers)]
        cls.test_pipeline_exp_a = train_classifier(
            exp_a_texts,
            cls.train_labels,
            model_path=cls.test_exp_a_path,
            max_features=100
        )

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary directory
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    def test_pipeline_structure(self):
        pipe = build_classifier_pipeline(max_features=50)
        self.assertIn("tfidf", pipe.named_steps)
        self.assertIn("clf", pipe.named_steps)

    def test_model_artifact_saved_and_loaded(self):
        self.assertTrue(self.test_model_path.is_file())
        self.assertTrue(is_classifier_available(self.test_model_path))

        loaded = load_classifier(self.test_model_path)
        self.assertIsNotNone(loaded)
        self.assertTrue(hasattr(loaded, "predict"))

    def test_format_context_text(self):
        q = "What is RPF?"
        ref = "RPF prevents broadcast duplicates."
        ans = "RPF prevents spoofing."

        self.assertEqual(format_context_text(ans, q, ref, mode="baseline"), "RPF prevents spoofing.")
        self.assertEqual(format_context_text(ans, q, ref, mode="question_answer"), "What is RPF? RPF prevents spoofing.")
        self.assertEqual(format_context_text(ans, q, ref, mode="question_ref_answer"), "What is RPF? RPF prevents broadcast duplicates. RPF prevents spoofing.")

    def test_predict_valid_answer(self):
        res = predict_correctness(
            student_answer="Accurate and valid definition of routing protocol.",
            model_path=self.test_model_path
        )
        self.assertTrue(res["available"])
        self.assertIn(res["predicted_label"], TARGET_CLASSES)
        self.assertIsNone(res["error"])
        self.assertIsInstance(res["probabilities"], dict)

        # Probabilities should sum to approximately 1.0
        prob_sum = sum(res["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_predict_with_context_exp_a(self):
        res = predict_correctness(
            student_answer="Accurate and valid definition of routing protocol.",
            question="Explain routing protocol.",
            model_path=self.test_exp_a_path,
            mode="question_answer"
        )
        self.assertTrue(res["available"])
        self.assertIn(res["predicted_label"], TARGET_CLASSES)
        self.assertIn("Exp A", res["model_type"])

    def test_predict_empty_or_whitespace_input(self):
        res_empty = predict_correctness("", model_path=self.test_model_path)
        self.assertIsNone(res_empty["predicted_label"])
        self.assertIn("cannot be empty", res_empty["error"])

        res_spaces = predict_correctness("   \n\t  ", model_path=self.test_model_path)
        self.assertIsNone(res_spaces["predicted_label"])
        self.assertIn("cannot be empty", res_spaces["error"])

    def test_predict_none_input(self):
        res_none = predict_correctness(None, model_path=self.test_model_path)
        self.assertIsNone(res_none["predicted_label"])
        self.assertIn("cannot be empty", res_none["error"])

    def test_missing_model_artifact(self):
        missing_path = self.test_dir / "does_not_exist.joblib"
        self.assertFalse(is_classifier_available(missing_path))

        res = predict_correctness(
            "Valid student answer text here.",
            model_path=missing_path
        )
        self.assertFalse(res["available"])
        self.assertIsNone(res["predicted_label"])
        self.assertEqual(res["probabilities"], {})
        self.assertIn("artifact not found", res["error"])
        self.assertNotIn("Correct", res["probabilities"])

    def test_load_nonexistent_model_returns_none(self):
        missing_path = self.test_dir / "ghost.joblib"
        loaded = load_classifier(missing_path)
        self.assertIsNone(loaded)


if __name__ == "__main__":
    unittest.main()
