
import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


class Settings:
    APP_NAME = "LLM Preprocessing Comparison System"
    APP_VERSION = "1.0.0"

    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # Project directories
    DATA_DIR = BASE_DIR / "data"
    DATABASE_DIR = DATA_DIR
    REPORTS_DIR = BASE_DIR / "reports"
    LOGS_DIR = BASE_DIR / "logs"

    # Database
    DATABASE_PATH = Path(
        os.getenv("DATABASE_PATH", str(DATA_DIR / "experiments.db"))
    )

    # Ollama configuration
    OLLAMA_BASE_URL = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    ).rstrip("/")

    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))

    # Default generation settings
    DEFAULT_TEMPERATURE = float(
        os.getenv("DEFAULT_TEMPERATURE", "0.2")
    )

    DEFAULT_TOP_P = float(
        os.getenv("DEFAULT_TOP_P", "0.9")
    )

    DEFAULT_MAX_TOKENS = int(
        os.getenv("DEFAULT_MAX_TOKENS", "512")
    )

    # Local Ollama models
    LLM_MODELS = {
        "smollm2": {
            "provider": "ollama",
            "model": "smollm2:360m",
        },
        "qwen": {
            "provider": "ollama",
            "model": "qwen2.5:0.5b",
        },
        "tinyllama": {
            "provider": "ollama",
            "model": "tinyllama:1.1b",
        },
        "llama": {
            "provider": "ollama",
            "model": "llama3.2:1b",
        },
        "gemma": {
            "provider": "ollama",
            "model": "gemma3:1b",
        },
    }


settings = Settings()


def create_project_directories() -> None:
    """Create directories required by the project."""
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
