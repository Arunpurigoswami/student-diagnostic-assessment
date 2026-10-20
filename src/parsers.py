"""
Parser and JSON cleaning utilities:
Handles stripping DeepSeek-R1 thinking traces (<think>...</think>),
extracting JSON code blocks, repairing common formatting flaws, and validating structure.
"""

import re
import json
from typing import Optional, Tuple


def clean_deepseek_output(raw_text: str) -> Tuple[str, Optional[str]]:
    """
    Split DeepSeek-R1 output into (cleaned_content, reasoning_thinking_trace).
    Strips out <think>...</think> blocks from the content.
    """
    if not raw_text:
        return "", None

    think_match = re.search(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL)
    thinking_trace = think_match.group(1).strip() if think_match else None

    # Remove the entire think block
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    return cleaned, thinking_trace


def repair_json_string(json_candidate: str) -> str:
    """
    Repair common JSON syntax flaws produced by small LLMs:
    - Trailing commas before } or ]
    - Python booleans/None (True, False, None)
    - Python set representations inside lists (e.g. {1, "text"})
    """
    text = json_candidate.strip()

    # Convert Python literals to JSON
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)

    # Convert Python sets like {1, "text"} inside JSON arrays to lists [1, "text"]
    text = re.sub(r"\{\s*([^{}:\"']+,\s*[^:]+?)\s*\}", r"[\1]", text)

    # Remove trailing commas before closing braces/brackets
    text = re.sub(r",\s*([\]}])", r"\1", text)

    return text


def parse_model_json(text: str) -> dict:
    """
    Robustly extract and parse a JSON dictionary from model output text.
    Handles markdown fences, unclosed fences, and embedded JSON objects.
    Raises ValueError if no valid JSON can be extracted.
    """
    if not text or not text.strip():
        raise ValueError("Empty model output text.")

    cleaned_text, _ = clean_deepseek_output(text)

    # Strategy 1: Look for ```json ... ``` or ``` ... ``` code blocks
    fence_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_text, flags=re.DOTALL)
    for candidate in fence_matches:
        try:
            return json.loads(candidate.strip())
        except Exception:
            try:
                return json.loads(repair_json_string(candidate))
            except Exception:
                pass

    # Strategy 2: Look for outermost curly braces { ... }
    brace_match = re.search(r"(\{.*\})", cleaned_text, flags=re.DOTALL)
    if brace_match:
        candidate = brace_match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return json.loads(repair_json_string(candidate))
            except Exception:
                pass

    # Strategy 3: Try raw direct load on cleaned text
    try:
        return json.loads(cleaned_text)
    except Exception:
        try:
            return json.loads(repair_json_string(cleaned_text))
        except Exception as err:
            raise ValueError(f"Could not parse valid JSON from model response: {err}")
