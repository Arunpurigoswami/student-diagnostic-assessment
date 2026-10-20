"""
Feedback module:
Generates targeted, actionable feedback conditioned on the localized error,
error category, and supporting evidence.
"""

from typing import Optional


def generate_feedback(diagnosis: dict,
                      score_info: dict,
                      evidence: Optional[str] = None,
                      raw_feedback: Optional[str] = None,
                      question: str = "",
                      reference_answer: str = "") -> str:
    """
    Generate actionable, constructive feedback tailored to the diagnosed error.
    """
    if raw_feedback and len(raw_feedback.strip()) > 15:
        return raw_feedback.strip()

    first_error_step = diagnosis.get("first_error_step")
    first_error_desc = diagnosis.get("first_error_description")
    error_type = diagnosis.get("error_type", "none")
    score = score_info.get("score", 0.0)

    if first_error_step is None or error_type == "none":
        return "Excellent work! Your answer is accurate and aligns with the expected solution."

    category_advice = {
        "conceptual": "Focus on the foundational definitions and core mechanics of this topic.",
        "methodological": "Re-evaluate the approach or technique you chose to address this problem.",
        "procedural": "Double-check the sequential execution and ensure all prerequisite steps are covered.",
        "computational": "Review your calculations and index variables for arithmetic or boundary discrepancies.",
        "logical_causal": "Inspect the cause-and-effect relationship between your premise and conclusion.",
        "question_misinterpretation": "Re-read the question carefully to identify exactly what is being requested.",
        "incomplete_reasoning": "Expand your explanation with the missing intermediate justifications."
    }

    advice = category_advice.get(error_type, "Review the reference solution to clarify this concept.")

    parts = [
        f"Diagnostic Assessment: A {error_type} error occurred at Step {first_error_step}."
    ]

    if first_error_desc:
        parts.append(f"Issue: {first_error_desc}")

    if evidence:
        parts.append(f'Referenced span in your answer: "{evidence}"')

    parts.append(f"Actionable guidance: {advice}")

    return "\n\n".join(parts)
