"""LLM client integration package for Ollama on Windows Host."""

from app.llm.ollama_client import OllamaClient, get_ollama_client

__all__ = [
    "OllamaClient",
    "get_ollama_client",
]
