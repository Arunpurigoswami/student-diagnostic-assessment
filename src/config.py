from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"

ERROR_TYPES = [
    "conceptual",
    "methodological",
    "procedural",
    "computational",
    "logical/causal",
    "question-misinterpretation",
    "incomplete reasoning",
]
