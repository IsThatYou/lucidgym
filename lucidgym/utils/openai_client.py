"""
Utility for creating OpenAI-compatible clients with support for multiple providers.
"""
import os
from openai import OpenAI


def get_openai_client(model: str | None = None) -> OpenAI:
    """
    Create an OpenAI client that works with OpenAI or OpenRouter.

    Uses OpenRouter if the model name starts with "openai/" prefix.
    Otherwise uses OpenAI directly.

    Args:
        model: Model name (e.g., "gpt-4" or "openai/gpt-4")

    Returns:
        OpenAI: Configured client instance
    """
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    # Use OpenRouter if model starts with "openai/" prefix
    if model and model.startswith("openai/"):
        if not openrouter_api_key:
            raise RuntimeError("Model uses 'openai/' prefix but OPENROUTER_API_KEY is not set")
        return OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        # Use OpenAI directly
        return OpenAI()
