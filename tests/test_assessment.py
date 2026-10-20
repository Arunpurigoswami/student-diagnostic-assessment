"""
Unit tests for Explainable NLP Student Diagnostic Assessment:
Tests validation, error localization, error classification, evidence verification,
scoring bounds, malformed JSON recovery, and server connection error handling
using mocked model responses.
"""

import unittest
from unittest.mock import MagicMock, patch
import openai

from src.reasoning import split_into_steps, extract_reasoning
from src.diagnosis import (
    validate_evidence_span,
    extract_grounded_evidence,
    localize_first_error,
    classify_error,
)
from src.scoring import score_answer
from src.feedback import generate_feedback
from src.parsers import clean_deepseek_output, parse_model_json, repair_json_string
from src.config import normalize_error_type, ERROR_TYPES
from src.pipeline import analyze_answer


class TestReasoningExtraction(unittest.TestCase):
    def test_split_numbered_steps(self):
        text = "1. First calculate latency. 2. Then multiply by bandwidth. 3. Output throughput."
        steps = split_into_steps(text)
        self.assertEqual(len(steps), 3)
        self.assertIn("First calculate latency.", steps[0])

    def test_split_prose_sentences(self):
        text = "Reverse Path Forwarding checks the incoming interface. If not optimal it drops the packet."
        steps = split_into_steps(text)
        self.assertEqual(len(steps), 2)

    def test_extract_reasoning_structure(self):
        text = "Step 1: Check link state.\nStep 2: Compute Dijkstra shortest path."
        structured = extract_reasoning(text)
        self.assertEqual(len(structured), 2)
        self.assertEqual(structured[0]["step_number"], 1)
        self.assertTrue(structured[0]["is_valid"])


class TestEvidenceVerification(unittest.TestCase):
    def setUp(self):
        self.student_answer = "Frame bursting concatenates small frames but it requires station buffering delay."

    def test_exact_evidence_span_preserved(self):
        candidate = "station buffering delay"
        span = validate_evidence_span(self.student_answer, candidate)
        self.assertEqual(span, "station buffering delay")

    def test_case_insensitive_evidence_span(self):
        candidate = "STATION BUFFERING DELAY"
        span = validate_evidence_span(self.student_answer, candidate)
        self.assertEqual(span, "station buffering delay")

    def test_never_fabricate_quotation(self):
        candidate = "station does not buffer any frames"
        span = validate_evidence_span(self.student_answer, candidate)
        self.assertIsNone(span)

    def test_extract_grounded_evidence_fallback(self):
        # Candidate is fabricated, but flawed step text is verbatim
        candidate = "completely made up hallucinated quote"
        flawed_step = "Frame bursting concatenates small frames"
        span = extract_grounded_evidence(self.student_answer, candidate, flawed_step)
        self.assertEqual(span, flawed_step)


class TestDiagnosisAndClassification(unittest.TestCase):
    def test_taxonomy_normalization(self):
        self.assertEqual(normalize_error_type("Conceptual"), "conceptual")
        self.assertEqual(normalize_error_type("methodological"), "methodological")
        self.assertEqual(normalize_error_type("procedural"), "procedural")
        self.assertEqual(normalize_error_type("computational"), "computational")
        self.assertEqual(normalize_error_type("logical/causal"), "logical_causal")
        self.assertEqual(normalize_error_type("question-misinterpretation"), "question_misinterpretation")
        self.assertEqual(normalize_error_type("incomplete reasoning"), "incomplete_reasoning")
        self.assertEqual(normalize_error_type(None), "none")

    def test_localize_first_error(self):
        steps = [
            {"step_number": 1, "text": "Step 1 text", "is_valid": True, "issue": None},
            {"step_number": 2, "text": "Step 2 text", "is_valid": True, "issue": None},
        ]
        res = localize_first_error(steps, first_error_step=2, error_description="Incorrect rule")
        self.assertEqual(res["first_error_step"], 2)
        self.assertFalse(res["reasoning_steps"][1]["is_valid"])
        self.assertEqual(res["reasoning_steps"][1]["issue"], "Incorrect rule")

    def test_classify_error_none_when_no_error(self):
        cat = classify_error("conceptual", first_error_step=None)
        self.assertEqual(cat, "none")


class TestScoringAndFeedback(unittest.TestCase):
    def test_score_clamping(self):
        res_high = score_answer("Q", "Rubric", [], {"first_error_step": 1}, raw_score=1.8)
        self.assertEqual(res_high["score"], 1.0)

        res_low = score_answer("Q", "Rubric", [], {"first_error_step": 1}, raw_score=-0.5)
        self.assertEqual(res_low["score"], 0.0)

        res_valid = score_answer("Q", "Rubric", [], {"first_error_step": 1}, raw_score=0.75)
        self.assertEqual(res_valid["score"], 0.75)

    def test_feedback_actionable(self):
        fb = generate_feedback(
            diagnosis={"first_error_step": 1, "first_error_description": "Confused RPF with encryption", "error_type": "conceptual"},
            score_info={"score": 0.0},
            evidence="help prevent IP address spoofing"
        )
        self.assertIn("conceptual", fb.lower())
        self.assertIn("help prevent IP address spoofing", fb)


class TestParsersAndJSONRepair(unittest.TestCase):
    def test_strip_deepseek_thinking_trace(self):
        raw = "<think>\nThinking about student reasoning...\n</think>\n```json\n{\"score\": 0.5}\n```"
        cleaned, trace = clean_deepseek_output(raw)
        self.assertEqual(trace, "Thinking about student reasoning...")
        self.assertNotIn("<think>", cleaned)

    def test_parse_model_json_with_fences(self):
        raw = "```json\n{\"first_error_step\": 1, \"error_type\": \"conceptual\"}\n```"
        data = parse_model_json(raw)
        self.assertEqual(data["first_error_step"], 1)
        self.assertEqual(data["error_type"], "conceptual")

    def test_repair_trailing_commas_and_python_literals(self):
        malformed = '{"valid": True, "items": [1, 2, ], "missing": None, }'
        repaired = repair_json_string(malformed)
        self.assertIn('"valid": true', repaired)
        self.assertIn('"missing": null', repaired)
        self.assertNotIn(", ]", repaired)


class TestPipelineWithMocks(unittest.TestCase):
    def test_missing_student_answer_validation(self):
        res = analyze_answer("What is IP?", "Internet Protocol", "1 pt", "")
        self.assertEqual(res["status"], "error")
        self.assertIn("cannot be empty", res["error"])

    def test_missing_question_validation(self):
        res = analyze_answer("", "Internet Protocol", "1 pt", "IP stands for Internet Protocol.")
        self.assertEqual(res["status"], "error")
        self.assertIn("cannot be empty", res["error"])

    def test_server_connection_error_handling(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=MagicMock(), message="Connection refused to 127.0.0.1:1234"
        )

        res = analyze_answer(
            question="What is RPF?",
            reference_answer="Reverse Path Forwarding reduces duplicates.",
            rubric="1 pt",
            student_answer="RPF prevents spoofing.",
            client=mock_client
        )
        self.assertEqual(res["status"], "error")
        self.assertIn("connection error", res["error"].lower())
        self.assertIn("LM Studio", res["feedback"])

    def test_malformed_json_fallback_recovery(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "<think>I think it is bad</think> This is totally invalid non-JSON output!"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        res = analyze_answer(
            question="What is RPF?",
            reference_answer="Reverse Path Forwarding reduces duplicates.",
            rubric="1 pt",
            student_answer="1. RPF prevents spoofing. 2. It checks normal path.",
            client=mock_client
        )
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["is_malformed_json_fallback"])
        self.assertEqual(len(res["reasoning_steps"]), 2)
        self.assertIn(res["error_type"], ERROR_TYPES)

    def test_successful_mocked_assessment(self):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = """<think>
Student confused purpose with spoofing prevention.
</think>
```json
{
  "reasoning_steps": [
    {"step_number": 1, "text": "1. Purpose: help prevent IP spoofing.", "is_valid": false, "issue": "Wrong purpose"},
    {"step_number": 2, "text": "2. RPF broadcasts to all stations.", "is_valid": true, "issue": null}
  ],
  "first_error_step": 1,
  "first_error_description": "Confused duplicate reduction with IP spoofing prevention.",
  "error_type": "conceptual",
  "evidence_span": "help prevent IP spoofing",
  "score": 0.25,
  "score_rationale": "Partially understood broadcast mechanism but wrong purpose.",
  "feedback": "Review the primary purpose of RPF in broadcast routing."
}
```"""
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        student_answer = "1. Purpose: help prevent IP spoofing. 2. RPF broadcasts to all stations."
        res = analyze_answer(
            question="What is RPF?",
            reference_answer="RPF reduces duplicates during broadcast.",
            rubric="1 pt",
            student_answer=student_answer,
            client=mock_client
        )

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["first_error_step"], 1)
        self.assertEqual(res["error_type"], "conceptual")
        self.assertEqual(res["evidence_span"], "help prevent IP spoofing")
        self.assertEqual(res["score"], 0.25)
        self.assertIn("Review the primary purpose", res["feedback"])
        self.assertEqual(res["raw_thinking"], "Student confused purpose with spoofing prevention.")


if __name__ == "__main__":
    unittest.main()
