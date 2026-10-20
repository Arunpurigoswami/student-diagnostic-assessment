"""
Scoring module:
Computes an estimated score fraction between 0.0 and 1.0 aligned with rubric
and diagnostic findings, accompanied by a rationale.
"""

from typing import Optional


def score_answer(question: str,
                 rubric: str,
                 reasoning_steps: list[dict],
                 diagnosis: dict,
                 raw_score: Optional[float] = None,
                 raw_rationale: Optional[str] = None) -> dict:
    """
    Return estimated score fraction between 0.0 and 1.0 with rationale.
    Guarantees score is bounded in [0.0, 1.0].
    """
    first_error_step = diagnosis.get("first_error_step")
    error_type = diagnosis.get("error_type", "none")

    resolved_score: float
    resolved_rationale: str

    if raw_score is not None:
        try:
            val = float(raw_score)
            resolved_score = max(0.0, min(1.0, round(val, 2)))
        except (ValueError, TypeError):
            resolved_score = 0.5
    else:
        # Fallback scoring based on reasoning steps
        if not reasoning_steps or first_error_step == 1:
            resolved_score = 0.0
        elif first_error_step is None or error_type == "none":
            resolved_score = 1.0
        else:
            total = len(reasoning_steps)
            valid_count = max(0, first_error_step - 1)
            resolved_score = round(valid_count / total, 2)

    if raw_rationale and str(raw_rationale).strip():
        resolved_rationale = str(raw_rationale).strip()
    else:
        if resolved_score == 1.0:
            resolved_rationale = "The student answer satisfies the rubric requirements without identified reasoning flaws."
        elif resolved_score == 0.0:
            resolved_rationale = f"A fundamental {error_type} error was identified at step {first_error_step or 1}, preventing credit."
        else:
            resolved_rationale = f"Partial credit awarded ({resolved_score}): correct reasoning until {error_type} error at step {first_error_step}."

    return {
        "score": resolved_score,
        "rationale": resolved_rationale
    }
