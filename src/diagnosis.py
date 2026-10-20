"""
Diagnosis module:
Handles error diagnosis, first-error localization, error classification,
and strict evidence span extraction.
"""

import re
from typing import Optional
from src.config import ERROR_TYPES, normalize_error_type


def validate_evidence_span(student_answer: str, candidate_span: Optional[str]) -> Optional[str]:
    """
    Validate and return an exact substring from the student's answer.
    Never fabricates a quotation.
    If the candidate span is not an authentic substring of student_answer, returns None.
    """
    if not candidate_span or not isinstance(candidate_span, str):
        return None

    candidate = candidate_span.strip().strip('"\'')
    if not candidate:
        return None

    # 1. Exact match check
    if candidate in student_answer:
        return candidate

    # 2. Case-insensitive exact match recovery
    lower_answer = student_answer.lower()
    lower_candidate = candidate.lower()
    start_idx = lower_answer.find(lower_candidate)
    if start_idx != -1:
        end_idx = start_idx + len(candidate)
        return student_answer[start_idx:end_idx]

    # 3. Handle ellipsis in quote (e.g., 'prefix ... suffix')
    if "..." in candidate:
        parts = [p.strip() for p in candidate.split("...") if p.strip()]
        if parts and parts[0] in student_answer:
            return parts[0]

    # Candidate span is not in student answer; do not fabricate
    return None


def extract_grounded_evidence(student_answer: str,
                              candidate_span: Optional[str] = None,
                              flawed_step_text: Optional[str] = None) -> Optional[str]:
    """
    Guarantee that the returned evidence span is an authentic, verbatim quote
    from student_answer. Falls back to verbatim flawed step text if present.
    """
    if not student_answer:
        return None

    # First attempt: validate candidate span
    valid_span = validate_evidence_span(student_answer, candidate_span)
    if valid_span:
        return valid_span

    # Fallback: check if the flawed step text exists in student_answer
    if flawed_step_text:
        step_candidate = validate_evidence_span(student_answer, flawed_step_text)
        if step_candidate:
            return step_candidate

    return None


def localize_first_error(reasoning_steps: list[dict],
                         first_error_step: Optional[int] = None,
                         error_description: Optional[str] = None) -> dict:
    """
    Identify and record the earliest supported error step in the reasoning chain.
    If no error is detected or step is invalid, returns first_error_step as None.
    """
    if not reasoning_steps:
        return {
            "first_error_step": None,
            "first_error_description": None,
            "reasoning_steps": []
        }

    valid_step_nums = {s.get("step_number") for s in reasoning_steps}

    # Validate that first_error_step exists in reasoning_steps
    resolved_step: Optional[int] = None
    if first_error_step is not None and first_error_step in valid_step_nums:
        resolved_step = first_error_step
    else:
        # Check if any step was pre-flagged as invalid
        for step in reasoning_steps:
            if step.get("is_valid") is False:
                resolved_step = step.get("step_number")
                break

    # Update reasoning steps state
    updated_steps = []
    error_found = False
    for step in reasoning_steps:
        step_copy = dict(step)
        num = step_copy.get("step_number")

        if resolved_step is not None and num == resolved_step:
            step_copy["is_valid"] = False
            if error_description and not step_copy.get("issue"):
                step_copy["issue"] = error_description
            error_found = True
        elif error_found:
            # Steps following the first error may be contaminated
            pass

        updated_steps.append(step_copy)

    return {
        "first_error_step": resolved_step,
        "first_error_description": error_description if resolved_step is not None else None,
        "reasoning_steps": updated_steps
    }


def classify_error(raw_category: Optional[str],
                   first_error_step: Optional[int] = None) -> str:
    """
    Map raw model classification to the standard canonical taxonomy:
    - conceptual
    - methodological
    - procedural
    - computational
    - logical_causal
    - question_misinterpretation
    - incomplete_reasoning
    - none
    """
    if first_error_step is None:
        return "none"

    normalized = normalize_error_type(raw_category)
    return normalized
