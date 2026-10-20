"""
Main diagnostic assessment pipeline:
Orchestrates reasoning extraction, diagnostic verification against reference & rubric,
first-error localization, error classification, authentic evidence quoting, scoring,
and feedback generation using a local LM Studio instance.
"""

from typing import Optional, Any
from openai import OpenAI
import openai

from src.config import (
    get_lm_studio_client,
    get_active_model_id,
    DEFAULT_BASE_URL,
    ERROR_TYPES,
)
from src.parsers import clean_deepseek_output, parse_model_json
from src.reasoning import extract_reasoning
from src.diagnosis import (
    localize_first_error,
    classify_error,
    extract_grounded_evidence,
)
from src.scoring import score_answer
from src.feedback import generate_feedback


def build_diagnostic_prompt(question: str,
                            reference_answer: str,
                            rubric: str,
                            student_answer: str) -> str:
    """Construct a rigorous prompt for step-by-step diagnostic reasoning assessment."""
    categories_str = "\n".join(f"- {cat}" for cat in ERROR_TYPES)
    rubric_text = rubric.strip() if rubric and rubric.strip() else "Award full credit for complete, accurate answers; deduct for conceptual flaws or omitted key mechanics."
    reference_text = reference_answer.strip() if reference_answer and reference_answer.strip() else "Not provided (evaluate based on domain correctness)."

    return f"""You are an educational NLP diagnostic assessment engine for grading free-form student answers.
Compare the student's answer against the question, reference answer, and rubric.

Question:
{question.strip()}

Reference Answer:
{reference_text}

Rubric:
{rubric_text}

Student Answer:
{student_answer.strip()}

Allowed error categories:
{categories_str}

Tasks:
1. Extract the explicit reasoning steps in the student answer.
2. Localize the FIRST error or omission (the earliest step where reasoning fails). If completely correct, first_error_step is null.
3. Classify the error into exactly one of the allowed categories.
4. Quote the exact evidence span from the student answer supporting the error. Never invent text not present in the student answer.
5. Provide a score fraction between 0.0 and 1.0 with rationale.
6. Provide constructive, actionable feedback.

Respond ONLY with a JSON object inside ```json ``` markdown code fences:
```json
{{
  "reasoning_steps": [
    {{"step_number": 1, "text": "...", "is_valid": true, "issue": null}},
    {{"step_number": 2, "text": "...", "is_valid": false, "issue": "..."}}
  ],
  "first_error_step": 2,
  "first_error_description": "Explanation of the first error",
  "error_type": "conceptual",
  "evidence_span": "exact quote from student answer",
  "score": 0.5,
  "score_rationale": "Explanation of score based on rubric",
  "feedback": "Actionable guidance on how to fix the error"
}}
```
"""


def analyze_answer(question: str,
                   reference_answer: str,
                   rubric: str,
                   student_answer: str,
                   base_url: Optional[str] = None,
                   model: Optional[str] = None,
                   client: Optional[Any] = None) -> dict:
    """
    Execute the complete Explainable NLP Student Diagnostic Assessment.
    Returns a structured dictionary with reasoning steps, first error,
    classified error type, authentic evidence span, score, and feedback.
    """
    # 1. Input Validation
    if not student_answer or not str(student_answer).strip():
        return {
            "status": "error",
            "error": "Student answer cannot be empty.",
            "reasoning_steps": [],
            "first_error_step": None,
            "first_error_description": None,
            "error_type": "none",
            "evidence_span": None,
            "score": 0.0,
            "score_rationale": "No student answer was provided.",
            "feedback": "Please enter a student answer to begin diagnostic assessment.",
            "model_used": None,
            "raw_thinking": None,
        }

    if not question or not str(question).strip():
        return {
            "status": "error",
            "error": "Question cannot be empty.",
            "reasoning_steps": [],
            "first_error_step": None,
            "first_error_description": None,
            "error_type": "none",
            "evidence_span": None,
            "score": 0.0,
            "score_rationale": "No question was provided.",
            "feedback": "Please enter a question for reference.",
            "model_used": None,
            "raw_thinking": None,
        }

    # 2. Configure LM Studio Client (local only; never cloud)
    active_base_url = base_url or DEFAULT_BASE_URL
    if client is None:
        try:
            client = get_lm_studio_client(active_base_url)
        except Exception as err:
            return {
                "status": "error",
                "error": f"Failed to initialize client for LM Studio at {active_base_url}: {err}",
                "reasoning_steps": extract_reasoning(student_answer),
                "first_error_step": None,
                "first_error_description": None,
                "error_type": "none",
                "evidence_span": None,
                "score": 0.0,
                "score_rationale": "Inference server unavailable.",
                "feedback": f"Could not connect to LM Studio at {active_base_url}. Please verify LM Studio local server is running.",
                "model_used": None,
                "raw_thinking": None,
            }

    # Resolve active model ID
    model_name = model
    if not model_name:
        try:
            model_name = get_active_model_id(client)
        except Exception:
            model_name = "deepseek-r1-distill-qwen-1.5b"

    # 3. Call local model
    prompt = build_diagnostic_prompt(question, reference_answer, rubric, student_answer)
    raw_content = ""
    thinking_trace: Optional[str] = None

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a specialized diagnostic NLP assessment engine for student reasoning. Output valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=900
        )
        raw_content = response.choices[0].message.content or ""
        _, thinking_trace = clean_deepseek_output(raw_content)
    except (openai.APIConnectionError, openai.APITimeoutError, Exception) as err:
        return {
            "status": "error",
            "error": f"LM Studio server connection error ({active_base_url}): {err}",
            "reasoning_steps": extract_reasoning(student_answer),
            "first_error_step": None,
            "first_error_description": None,
            "error_type": "none",
            "evidence_span": None,
            "score": 0.0,
            "score_rationale": "LM Studio server connection failed.",
            "feedback": f"Could not connect to LM Studio at {active_base_url}. Ensure LM Studio is open and the local server is started on port 1234.",
            "model_used": model_name,
            "raw_thinking": None,
        }

    # 4. Parse model JSON with fallback handling
    parsed: dict = {}
    is_malformed = False

    try:
        parsed = parse_model_json(raw_content)
    except ValueError:
        is_malformed = True
        parsed = {}

    # Extract or fallback reasoning steps
    raw_steps = parsed.get("reasoning_steps")
    if isinstance(raw_steps, list) and len(raw_steps) > 0 and isinstance(raw_steps[0], dict):
        steps = []
        for i, s in enumerate(raw_steps, start=1):
            steps.append({
                "step_number": s.get("step_number", i),
                "text": str(s.get("text", "")).strip(),
                "is_valid": bool(s.get("is_valid", True)),
                "issue": s.get("issue")
            })
    else:
        steps = extract_reasoning(student_answer)

    # 5. Localize first error
    raw_first_step = parsed.get("first_error_step")
    raw_first_desc = parsed.get("first_error_description")
    loc_result = localize_first_error(steps, raw_first_step, raw_first_desc)

    # 6. Classify error type
    first_step_num = loc_result.get("first_error_step")
    error_type = classify_error(parsed.get("error_type"), first_step_num)

    # 7. Extract and strictly verify authentic evidence span (Never fabricate)
    flawed_text = None
    if first_step_num:
        for s in steps:
            if s.get("step_number") == first_step_num:
                flawed_text = s.get("text")
                break

    candidate_span = parsed.get("evidence_span")
    verified_evidence = extract_grounded_evidence(
        student_answer=student_answer,
        candidate_span=candidate_span,
        flawed_step_text=flawed_text
    )

    # 8. Score answer
    score_info = score_answer(
        question=question,
        rubric=rubric,
        reasoning_steps=loc_result["reasoning_steps"],
        diagnosis={
            "first_error_step": first_step_num,
            "error_type": error_type
        },
        raw_score=parsed.get("score"),
        raw_rationale=parsed.get("score_rationale")
    )

    # 9. Generate feedback
    feedback_text = generate_feedback(
        diagnosis={
            "first_error_step": first_step_num,
            "first_error_description": loc_result.get("first_error_description"),
            "error_type": error_type
        },
        score_info=score_info,
        evidence=verified_evidence,
        raw_feedback=parsed.get("feedback"),
        question=question,
        reference_answer=reference_answer
    )

    return {
        "status": "success",
        "reasoning_steps": loc_result["reasoning_steps"],
        "first_error_step": loc_result["first_error_step"],
        "first_error_description": loc_result["first_error_description"],
        "error_type": error_type,
        "evidence_span": verified_evidence,
        "score": score_info["score"],
        "score_rationale": score_info["rationale"],
        "feedback": feedback_text,
        "model_used": model_name,
        "raw_thinking": thinking_trace,
        "is_malformed_json_fallback": is_malformed
    }
