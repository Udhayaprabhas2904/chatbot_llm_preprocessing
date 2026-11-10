from dataclasses import asdict, dataclass
from typing import Optional

from config.settings import settings


@dataclass(frozen=True)
class ModelInfo:
    """Metadata describing an LLM and its provider."""

    family: str
    model_tag: str
    provider: str = "ollama"
    description: str = ""
    enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class ModelRegistry:
    """Registry for the five local Ollama models."""

    def __init__(self) -> None:
        self._models: dict[str, ModelInfo] = {}

        descriptions = {
            "smollm2": "SmolLM2 360M running locally through Ollama.",
            "qwen": "Qwen 2.5 0.5B running locally through Ollama.",
            "tinyllama": "TinyLlama 1.1B running locally through Ollama.",
            "llama": "Llama 3.2 1B running locally through Ollama.",
            "gemma": "Gemma 3 1B running locally through Ollama.",
        }

        for family, config in settings.LLM_MODELS.items():
            if isinstance(config, dict):
                model_tag = config.get("model", "")
                provider = config.get("provider", "ollama")
            else:
                model_tag = config
                provider = "ollama"

            self.register_model(
                ModelInfo(
                    family=family,
                    model_tag=model_tag,
                    provider=provider,
                    description=descriptions.get(
                        family.lower(),
                        f"{family.title()} language model.",
                    ),
                )
            )

    def register_model(self, model: ModelInfo) -> None:
        family = model.family.strip().lower()
        model_tag = model.model_tag.strip()
        provider = model.provider.strip().lower()

        if not family:
            raise ValueError("Model family cannot be empty.")

        if not model_tag:
            raise ValueError("Model tag cannot be empty.")

        if not provider:
            raise ValueError("Model provider cannot be empty.")

        self._models[family] = ModelInfo(
            family=family,
            model_tag=model_tag,
            provider=provider,
            description=model.description,
            enabled=model.enabled,
        )

    def get_model(self, family: str) -> ModelInfo:
        if not isinstance(family, str) or not family.strip():
            raise ValueError("Model family cannot be empty.")

        key = family.strip().lower()

        if key not in self._models:
            available = ", ".join(sorted(self._models))
            raise KeyError(
                f"Unknown model family '{family}'. "
                f"Available families: {available}"
            )

        return self._models[key]

    def get_model_tag(self, family: str) -> str:
        return self.get_model(family).model_tag

    def get_provider(self, family: str) -> str:
        return self.get_model(family).provider

    def list_models(self, enabled_only: bool = False) -> list[dict]:
        models = [
            model
            for model in self._models.values()
            if not enabled_only or model.enabled
        ]

        return [
            model.to_dict()
            for model in sorted(models, key=lambda item: item.family)
        ]

    def list_families(self, enabled_only: bool = False) -> list[str]:
        return [
            model["family"]
            for model in self.list_models(enabled_only)
        ]

    def enable_model(self, family: str) -> None:
        model = self.get_model(family)

        self._models[model.family] = ModelInfo(
            **{**model.to_dict(), "enabled": True}
        )

    def disable_model(self, family: str) -> None:
        model = self.get_model(family)

        self._models[model.family] = ModelInfo(
            **{**model.to_dict(), "enabled": False}
        )

    def is_registered(self, family: str) -> bool:
        if not isinstance(family, str):
            return False

        return family.strip().lower() in self._models

    def find_by_tag(self, model_tag: str) -> Optional[ModelInfo]:
        if not isinstance(model_tag, str):
            return None

        target = model_tag.strip().lower()

        for model in self._models.values():
            if model.model_tag.lower() == target:
                return model

        return None


model_registry = ModelRegistry()