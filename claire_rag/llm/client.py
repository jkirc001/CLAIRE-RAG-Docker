"""LLM client with configuration management."""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

# Load .env file if it exists
load_dotenv()


class LLMClient:
    """
    Centralized LLM client with configuration management.

    Loads configuration from:
    - .env file (environment variables)
    - config/settings.yaml
    - config/models.yaml

    Enforces model selection rules based on environment mode.
    """

    def __init__(self, config_dir: Path | None = None):
        """
        Initialize LLM client with configuration.

        Args:
            config_dir: Optional path to config directory. Defaults to project root / config
        """
        if config_dir is None:
            # Assume config is in project root
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"

        self.config_dir = Path(config_dir)

        # Load .env file from config directory if it exists
        env_file = self.config_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)

        self.settings = self._load_settings()
        self.models_config = self._load_models_config()
        self.env_mode = os.getenv("CLAIRE_ENV", "development")

        # Resolve model based on rules
        self.model = self._resolve_model()
        self.temperature = self.settings.get("llm", {}).get("temperature", 0.2)
        self.max_tokens = self.settings.get("llm", {}).get("max_tokens", 2048)

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY not found in environment. "
                "LLM calls will fail unless API key is provided."
            )
        self.client = OpenAI(api_key=api_key) if api_key else None

        # Log model selection
        if self.env_mode == "evaluation":
            logger.info(
                f"Evaluation mode is active. Overriding LLM model to {self.model}."
            )

    def _load_settings(self) -> dict[str, Any]:
        """Load settings.yaml."""
        settings_path = self.config_dir / "settings.yaml"
        if not settings_path.exists():
            raise FileNotFoundError(f"Settings file not found: {settings_path}")

        with open(settings_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _load_models_config(self) -> dict[str, Any]:
        """Load models.yaml."""
        models_path = self.config_dir / "models.yaml"
        if not models_path.exists():
            raise FileNotFoundError(f"Models config file not found: {models_path}")

        with open(models_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _resolve_model(self) -> str:
        """
        Resolve the model to use based on environment and configuration.

        Returns:
            Model name to use

        Raises:
            ValueError: If model is not in allowed_models list
        """
        # Check environment mode
        if self.env_mode == "evaluation":
            evaluation_model = self.models_config.get("evaluation_model")
            if evaluation_model:
                return evaluation_model
            # Fallback to gpt-4o if not specified
            return "gpt-4o"

        # Development mode: use configured model
        configured_model = self.settings.get("llm", {}).get("model", "gpt-4o-mini")

        # Validate model is allowed
        allowed_models = self.models_config.get("allowed_models", [])
        if configured_model not in allowed_models:
            raise ValueError(
                f"Model '{configured_model}' is not in allowed_models list: {allowed_models}"
            )

        return configured_model

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        return_usage: bool = False,
    ) -> str | tuple[str, dict]:
        """
        Generate text using the configured LLM.

        Args:
            prompt: Input prompt
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            return_usage: If True, return tuple of (text, usage_dict)

        Returns:
            Generated text, or tuple of (text, usage_dict) if return_usage=True

        Raises:
            RuntimeError: If OpenAI client is not initialized
        """
        if self.client is None:
            raise RuntimeError(
                "OpenAI client not initialized. Set OPENAI_API_KEY environment variable."
            )

        system_message = "You are an expert cybersecurity knowledge assistant."
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )

        answer = response.choices[0].message.content or ""

        if return_usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": (
                    response.usage.completion_tokens if response.usage else 0
                ),
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
            return answer, usage

        return answer

    def generate_stub(
        self, prompt: str, return_usage: bool = False
    ) -> str | tuple[str, dict]:
        """
        Stub implementation for testing without API calls.

        Args:
            prompt: Input prompt
            return_usage: If True, return tuple of (text, usage_dict)

        Returns:
            Stub response, or tuple of (text, usage_dict) if return_usage=True
        """
        stub_response = f"[STUB] Response to: {prompt[:50]}..."

        if return_usage:
            # Return zero usage for stub
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            return stub_response, usage

        return stub_response


def get_llm_client(config_dir: Path | None = None, use_stub: bool = False) -> LLMClient:
    """
    Factory function to get an LLM client instance.

    Args:
        config_dir: Optional path to config directory
        use_stub: If True, return a client that uses stub mode (for testing)

    Returns:
        LLMClient instance
    """
    client = LLMClient(config_dir=config_dir)
    if use_stub:
        # Replace generate method with stub
        client.generate = client.generate_stub
    return client
