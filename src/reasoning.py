"""
Reasoning extraction module:
Extracts the reasoning steps explicitly present in the student's answer.
"""

import re


def split_into_steps(student_answer: str) -> list[str]:
    """
    Decompose student answer into ordered, non-empty explicit steps
    using numbered lists, bullet points, or sentence boundaries.
    Avoids splitting on digit-period headers like '1. '.
    """
    if not student_answer or not student_answer.strip():
        return []

    text = student_answer.strip()

    # 1. Match numbered items like '1. ', '2) ', 'Step 1: '
    parts = re.split(r"(?:\s+|\n|^)(?=(?:Step\s*\d+:?|\d+[\.\)]|\(?\d+\))\s+)", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        return parts

    # 2. Try newline separation if multiple structured lines exist
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and all(len(line) > 5 for line in lines):
        return lines

    # 3. Sentence segmentation: ensure not splitting on single digits like '1. ' or abbreviations
    sentences = re.split(r"(?<!\b\d)(?<=[.!?])\s+(?=[A-Z0-9])", text)
    cleaned = [s.strip() for s in sentences if s.strip()]
    return cleaned if cleaned else [text]


def extract_reasoning(student_answer: str) -> list[dict]:
    """
    Extract the reasoning steps explicitly present in the student's answer.
    Returns a list of structured step dictionaries:
    [
        {
            "step_number": 1,
            "text": "...",
            "is_valid": True,
            "issue": None
        },
        ...
    ]
    """
    raw_steps = split_into_steps(student_answer)
    structured_steps = []
    for idx, step_text in enumerate(raw_steps, start=1):
        structured_steps.append({
            "step_number": idx,
            "text": step_text,
            "is_valid": True,
            "issue": None,
        })
    return structured_steps
