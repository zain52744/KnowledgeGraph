import logging
import time
import asyncio
from typing import Any, Callable, Optional
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    from langfuse.decorators import langfuse_context
    LANGFUSE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    LANGFUSE_AVAILABLE = False
    langfuse_context = None
    logger.warning("Langfuse not available, tracing disabled")


class LangfuseManager:

    def __init__(self) -> None:
        self._client = None
        self._enabled = False
        self._callback_handler = None

    def init(self) -> bool:
        try:
            from config import settings
            if not settings.langfuse_public_key or not settings.langfuse_secret_key:
                logger.warning("Langfuse keys not configured, monitoring disabled")
                return False

            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            self._enabled = True
            logger.info("Langfuse initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse: {e}")
            self._enabled = False
            return False

    def flush(self) -> None:
        if self._callback_handler and self._enabled:
            try:
                self._callback_handler.flush()
            except Exception as e:
                logger.warning(f"Failed to flush Langfuse traces: {e}")

    def get_callback_handler(self):
        if not self._enabled:
            return None
        if self._callback_handler is not None:
            return self._callback_handler

        from config import settings
        try:
            from langfuse.callback import CallbackHandler
            self._callback_handler = CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            logger.info("Langfuse CallbackHandler created successfully")
            return self._callback_handler
        except ImportError:
            pass
        try:
            from langfuse.langchain import CallbackHandler
            self._callback_handler = CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            logger.info("Langfuse CallbackHandler created successfully")
            return self._callback_handler
        except Exception as e:
            logger.warning(f"Langfuse LangChain integration not available: {e}")
            return None

    @property
    def enabled(self) -> bool:
        return self._enabled


langfuse_manager = LangfuseManager()
