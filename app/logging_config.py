"""Structured Application Logging Configuration.

Provides structured logging with execution timings and context tags,
omitting sensitive raw document contents from logs.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

from app.config import settings


def configure_logging() -> logging.Logger:
    """Initialize and configure root application logger."""
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Custom format with timestamp, level, module, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("rag_app")
    logger.setLevel(level)

    # Avoid duplicate handlers if re-executed
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = configure_logging()


@contextmanager
def timed_operation(
    operation_name: str,
    extra_context: Optional[dict[str, Any]] = None,
    log_level: int = logging.INFO,
) -> Generator[dict[str, Any], None, None]:
    """Context manager to measure and log operation latency in milliseconds."""
    context: dict[str, Any] = extra_context.copy() if extra_context else {}
    start_time = time.perf_counter()
    logger.log(log_level, f"START: {operation_name} | {context}")
    try:
        yield context
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error(
            f"FAILED: {operation_name} after {elapsed_ms:.2f}ms | Error: {exc} | {context}",
            exc_info=True,
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        context["elapsed_ms"] = elapsed_ms
        logger.log(
            log_level,
            f"COMPLETED: {operation_name} in {elapsed_ms:.2f}ms | {context}",
        )
