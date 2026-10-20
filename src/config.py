import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"

# Load local environment configuration if present
load_dotenv(PROJECT_ROOT / ".env")

# Canonical error taxonomy requested by the project specifications
ERROR_TYPES = [
    "conceptual",
    "methodological",
    "procedural",
    "computational",
    "logical_causal",
    "question_misinterpretation",
    "incomplete_reasoning",
    "none",
]

DEFAULT_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1"))
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "deepseek-r1-distill-qwen-1.5b")


def normalize_error_type(raw_type: str | None) -> str:
    """Normalize model output error type to standard taxonomy keys."""
    if not raw_type:
        return "none"
    clean = str(raw_type).strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    mapping = {
        "conceptual": "conceptual",
        "methodological": "methodological",
        "procedural": "procedural",
        "computational": "computational",
        "logical_causal": "logical_causal",
        "logical": "logical_causal",
        "causal": "logical_causal",
        "question_misinterpretation": "question_misinterpretation",
        "misinterpretation": "question_misinterpretation",
        "incomplete_reasoning": "incomplete_reasoning",
        "incomplete": "incomplete_reasoning",
        "none": "none",
        "no_error": "none",
        "correct": "none",
    }
    return mapping.get(clean, "none" if clean in ("none", "null") else "conceptual")


def get_lm_studio_client(base_url: str | None = None) -> OpenAI:
    """Instantiate OpenAI client configured strictly for local LM Studio."""
    url = (base_url or DEFAULT_BASE_URL).rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    # Use dummy local API key; never call OpenAI cloud
    return OpenAI(base_url=url, api_key="lm-studio", timeout=60.0)


def fetch_available_models(base_url: str | None = None) -> list[str]:
    """Query LM Studio /v1/models endpoint for available local models."""
    try:
        client = get_lm_studio_client(base_url)
        res = client.models.list()
        return [m.id for m in res.data]
    except Exception:
        return []


def get_active_model_id(client: OpenAI, preferred: str | None = None) -> str:
    """
    Resolve active model ID dynamically from /v1/models.
    Filters out embedding models to prioritize the generative reasoning model.
    """
    try:
        models = [m.id for m in client.models.list().data]
        if preferred and preferred in models:
            return preferred
        # Filter out embedding models
        generative = [m for m in models if "embed" not in m.lower()]
        if generative:
            return generative[0]
        if models:
            return models[0]
    except Exception:
        pass
    return preferred or DEFAULT_MODEL
