"""Ollama HTTP Client for Windows Host LLM Runtime.

Communicates with Ollama running natively on Windows host over the network API.
Supports health probes, batch generation, streaming generation, and timing tracking.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Generator, List, Optional, Tuple

from app.config import settings
from app.logging_config import logger, timed_operation


class OllamaClient:
    """Client for Ollama REST API running on the Windows host."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        mock_mode: bool = False,
    ) -> None:
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT_SECONDS
        self.mock_mode = mock_mode

    def health_check(self) -> Dict[str, Any]:
        """Verify Ollama service is reachable on Windows host and model is loaded."""
        if self.mock_mode:
            return {
                "status": "connected",
                "mode": "mock",
                "url": self.base_url,
                "model": self.model,
                "model_available": True,
                "available_models": [self.model],
            }

        url = f"{self.base_url}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "RAG-Chatbot-Client"})
        try:
            with urllib.request.urlopen(req, timeout=5.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    models = [m.get("name") for m in data.get("models", [])]
                    model_loaded = any(
                        self.model in m or m.startswith(self.model) for m in models
                    )
                    return {
                        "status": "connected",
                        "url": self.base_url,
                        "model": self.model,
                        "model_available": model_loaded,
                        "available_models": models,
                    }
                return {
                    "status": "error",
                    "url": self.base_url,
                    "error": f"Unexpected HTTP status {response.status}",
                }
        except Exception as exc:
            return {
                "status": "disconnected",
                "url": self.base_url,
                "error": f"Cannot connect to Ollama at {self.base_url}: {exc}",
                "hint": "Ensure OLLAMA_HOST=0.0.0.0:11434 is set on Windows and host IP is correct.",
            }

    def list_models(self) -> List[Dict[str, Any]]:
        """Retrieve list of available models installed in Ollama."""
        if self.mock_mode:
            return [{"name": self.model, "size": 1321098329, "details": {"parameter_size": "1.2B"}}]

        url = f"{self.base_url}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "RAG-Chatbot-Client"})
        try:
            with urllib.request.urlopen(req, timeout=5.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    return data.get("models", [])
                return []
        except Exception as exc:
            logger.warning(f"Failed to list Ollama models: {exc}")
            return []

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        return_meta: bool = False,
    ) -> Tuple[Any, ...]:
        """Send prompt to Ollama on Windows host and return generated response with latency.

        Returns:
            Tuple of (response_text, generation_time_ms) or
            (response_text, generation_time_ms, metadata_dict) if return_meta=True
        """
        active_model = model or self.model
        if self.mock_mode:
            gen_time_ms = 45.0
            display_name = "Llama 3.2 1B" if active_model in ("llama3.2:1b", "Llama 3.2 1B") else active_model
            mock_ans = (
                f"[Mock {display_name}] Generated answer based on context for query."
            )
            mock_meta = {
                "model": active_model,
                "eval_count": 15,
                "prompt_eval_count": 50,
                "tokens_per_second": 333.3,
            }
            if return_meta:
                return mock_ans, gen_time_ms, mock_meta
            return mock_ans, gen_time_ms

    @staticmethod
    def _clean_response(text: str) -> str:
        """Strip model-specific special tokens, leaked prompt instructions, and repetition loops."""
        if not text:
            return ""

        # 1. Strip special tokens returned by code/completion models like StarCoder
        special_tokens = [
            "<|end_of_text|>",
            "<end_of_text>",
            "<|endoftext|>",
            "<|end_of_answer|>",
            "|end_of_answer|>",
            "<file_sep>",
            "<fim_prefix>",
            "<fim_suffix>",
            "<fim_middle>",
        ]
        for tok in special_tokens:
            text = text.replace(tok, "")

        # 2. Cut off at stop markers if any slipped through
        stop_markers = [
            "\nUser Question:",
            "\nReference Context:",
            "\nContext:",
            "\nQuestion:",
            "\nUSER:",
            "\nSYSTEM:",
        ]
        for marker in stop_markers:
            if marker in text:
                text = text.split(marker)[0]

        # 3. Strip leaked prompt instructions
        prompt_leaks = [
            "Provide a more detailed answer:",
            "Provide a concise synthesized summary answer:",
            "Provide a concise synthesized answer:",
        ]
        for leak in prompt_leaks:
            text = text.replace(leak, "")

        text = text.strip()

        # 4. Strip leading answer labels
        for prefix in ["Synthesized Answer:", "Answer:", "Assistant:", "SYSTEM:", "USER:"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # 5. Deduplicate identical adjacent lines to break repetition loops
        lines = [line.strip() for line in text.split("\n")]
        deduped: List[str] = []
        for line in lines:
            if not deduped or line != deduped[-1]:
                deduped.append(line)

        return "\n".join(deduped).strip()

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        return_meta: bool = False,
    ) -> Tuple[Any, ...]:
        """Send prompt to Ollama on Windows host and return generated response with latency.

        Returns:
            Tuple of (response_text, generation_time_ms) or
            (response_text, generation_time_ms, metadata_dict) if return_meta=True
        """
        active_model = model or self.model
        if self.mock_mode:
            gen_time_ms = 45.0
            display_name = "Llama 3.2 1B" if active_model in ("llama3.2:1b", "Llama 3.2 1B") else active_model
            mock_ans = (
                f"[Mock {display_name}] Generated answer based on context for query."
            )
            mock_meta = {
                "model": active_model,
                "eval_count": 15,
                "prompt_eval_count": 50,
                "tokens_per_second": 333.3,
            }
            if return_meta:
                return mock_ans, gen_time_ms, mock_meta
            return mock_ans, gen_time_ms

        default_opts = {
            "temperature": settings.LLM_TEMPERATURE,
            "top_p": 0.9,
            "num_ctx": 2048,
            "num_predict": settings.LLM_NUM_PREDICT,
            "stop": [
                "\nUser Question:",
                "\nReference Context:",
                "\nContext:",
                "\nUSER:",
                "\nSYSTEM:",
                "<file_sep>",
                "<|endoftext|>",
                "<end_of_text>",
                "<|end_of_text>",
                "<|end_of_answer|>",
                "|end_of_answer|>",
            ],
        }
        if any(m in active_model.lower() for m in ("starcoder", "codestral")):
            default_opts["repeat_penalty"] = 1.15
        if options:
            default_opts.update(options)

        payload = {
            "model": active_model,
            "prompt": prompt,
            "stream": False,
            "options": default_opts,
        }
        if system:
            payload["system"] = system

        data_bytes = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/api/generate"
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t_start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                duration_ms = (time.perf_counter() - t_start) * 1000.0
                if response.status == 200:
                    resp_json = json.loads(response.read().decode("utf-8"))
                    text = self._clean_response(resp_json.get("response", ""))

                    eval_count = resp_json.get("eval_count", 0)
                    prompt_eval_count = resp_json.get("prompt_eval_count", 0)
                    eval_duration_ns = resp_json.get("eval_duration", 0)
                    tokens_per_sec = (
                        round(eval_count / (eval_duration_ns / 1e9), 2)
                        if eval_duration_ns > 0
                        else 0.0
                    )
                    meta = {
                        "model": active_model,
                        "eval_count": eval_count,
                        "prompt_eval_count": prompt_eval_count,
                        "eval_duration_ms": round(eval_duration_ns / 1e6, 2),
                        "load_duration_ms": round(resp_json.get("load_duration", 0) / 1e6, 2),
                        "total_duration_ms": round(resp_json.get("total_duration", 0) / 1e6, 2),
                        "tokens_per_second": tokens_per_sec,
                    }
                    logger.info(
                        f"Ollama generation ({active_model}) completed in {duration_ms:.2f}ms "
                        f"({len(text)} chars, {eval_count} tokens, {tokens_per_sec} tok/s)."
                    )
                    if return_meta:
                        return text, round(duration_ms, 2), meta
                    return text, round(duration_ms, 2)
                raise RuntimeError(
                    f"Ollama returned HTTP status {response.status}: {response.read().decode('utf-8')}"
                )
        except Exception as exc:
            duration_ms = (time.perf_counter() - t_start) * 1000.0
            logger.error(f"Ollama generation request failed after {duration_ms:.2f}ms: {exc}")
            raise RuntimeError(f"Failed to communicate with Ollama on Windows host: {exc}") from exc

    def chat(
        self,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        return_meta: bool = False,
    ) -> Tuple[Any, ...]:
        """Send structured chat messages to Ollama /api/chat."""
        active_model = model or self.model
        if self.mock_mode:
            gen_time_ms = 45.0
            display_name = "Llama 3.2 1B" if active_model in ("llama3.2:1b", "Llama 3.2 1B") else active_model
            mock_ans = (
                f"[Mock {display_name}] Generated answer based on context for query."
            )
            mock_meta = {
                "model": active_model,
                "eval_count": 15,
                "prompt_eval_count": 50,
                "tokens_per_second": 333.3,
            }
            if return_meta:
                return mock_ans, gen_time_ms, mock_meta
            return mock_ans, gen_time_ms

        default_opts = {
            "temperature": settings.LLM_TEMPERATURE,
            "top_p": 0.9,
            "num_ctx": 2048,
            "num_predict": settings.LLM_NUM_PREDICT,
            "stop": [
                "\nUser Question:",
                "\nReference Context:",
                "\nContext:",
                "\nUSER:",
                "\nSYSTEM:",
                "<file_sep>",
                "<|endoftext|>",
                "<end_of_text>",
                "<|end_of_text>",
                "<|end_of_answer|>",
                "|end_of_answer|>",
            ],
        }
        if any(m in active_model.lower() for m in ("starcoder", "codestral")):
            default_opts["repeat_penalty"] = 1.15
        if options:
            default_opts.update(options)

        payload = {
            "model": active_model,
            "messages": messages,
            "stream": False,
            "options": default_opts,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/api/chat"
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t_start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                duration_ms = (time.perf_counter() - t_start) * 1000.0
                if response.status == 200:
                    resp_json = json.loads(response.read().decode("utf-8"))
                    text = self._clean_response(resp_json.get("message", {}).get("content", ""))

                    eval_count = resp_json.get("eval_count", 0)
                    prompt_eval_count = resp_json.get("prompt_eval_count", 0)
                    eval_duration_ns = resp_json.get("eval_duration", 0)
                    tokens_per_sec = (
                        round(eval_count / (eval_duration_ns / 1e9), 2)
                        if eval_duration_ns > 0
                        else 0.0
                    )
                    meta = {
                        "model": active_model,
                        "eval_count": eval_count,
                        "prompt_eval_count": prompt_eval_count,
                        "eval_duration_ms": round(eval_duration_ns / 1e6, 2),
                        "load_duration_ms": round(resp_json.get("load_duration", 0) / 1e6, 2),
                        "total_duration_ms": round(resp_json.get("total_duration", 0) / 1e6, 2),
                        "tokens_per_second": tokens_per_sec,
                    }
                    logger.info(
                        f"Ollama chat ({active_model}) completed in {duration_ms:.2f}ms "
                        f"({len(text)} chars, {eval_count} tokens, {tokens_per_sec} tok/s)."
                    )
                    if return_meta:
                        return text, round(duration_ms, 2), meta
                    return text, round(duration_ms, 2)
                raise RuntimeError(
                    f"Ollama returned HTTP status {response.status}: {response.read().decode('utf-8')}"
                )
        except Exception as exc:
            duration_ms = (time.perf_counter() - t_start) * 1000.0
            logger.error(f"Ollama chat request failed after {duration_ms:.2f}ms: {exc}")
            raise RuntimeError(f"Failed to communicate with Ollama on Windows host: {exc}") from exc

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream generated tokens from Ollama on Windows host."""
        active_model = model or self.model
        if self.mock_mode:
            tokens = ["This ", "is ", "a ", "mock ", "streamed ", "response."]
            for tok in tokens:
                yield tok
            return

        default_opts = {
            "temperature": settings.LLM_TEMPERATURE,
            "top_p": 0.9,
            "num_ctx": 2048,
            "num_predict": settings.LLM_NUM_PREDICT,
            "stop": [
                "\nUser Question:",
                "\nReference Context:",
                "\nContext:",
                "\nUSER:",
                "\nSYSTEM:",
                "<file_sep>",
                "<|endoftext|>",
                "<end_of_text>",
                "<|end_of_text|>",
                "<|end_of_answer|>",
                "|end_of_answer|>",
            ],
        }
        if any(m in active_model.lower() for m in ("starcoder", "codestral")):
            default_opts["repeat_penalty"] = 1.15
        if options:
            default_opts.update(options)

        payload = {
            "model": active_model,
            "prompt": prompt,
            "stream": True,
            "options": default_opts,
        }
        if system:
            payload["system"] = system

        data_bytes = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/api/generate"
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                for line in response:
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
        except Exception as exc:
            logger.error(f"Error during Ollama token streaming: {exc}")
            yield f"\n[Generation error: {exc}]"


_ollama_client_instance: Optional[OllamaClient] = None


def get_ollama_client(mock_mode: bool = False) -> OllamaClient:
    """Retrieve or initialize singleton OllamaClient."""
    global _ollama_client_instance
    if _ollama_client_instance is None:
        _ollama_client_instance = OllamaClient(mock_mode=mock_mode)
    return _ollama_client_instance
