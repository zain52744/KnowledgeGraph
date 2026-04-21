import logging
import os
import time
import asyncio
import json
from typing import Any, Callable, Optional
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Global langfuse client instance
_langfuse_client = None
_langfuse_enabled = False
_callback_handler = None

# Try to import langfuse context
try:
    from langfuse.decorators import langfuse_context
    LANGFUSE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    LANGFUSE_AVAILABLE = False
    langfuse_context = None
    logger.warning("Langfuse not available, tracing disabled")


def init_langfuse():
   
    global _langfuse_client, _langfuse_enabled
    
    try:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            logger.warning("Langfuse keys not configured, monitoring disabled")
            _langfuse_enabled = False
            return False

        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

        _langfuse_enabled = True
        logger.info("Langfuse initialized successfully")
        return True

    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse: {e}")
        _langfuse_enabled = False
        return False


def trace_llm_call(name: str = None):
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            if not LANGFUSE_AVAILABLE or not langfuse_context:
                return await func(*args, **kwargs)
            trace_name = name or func.__name__
            start_time = time.time()
            try:
                with langfuse_context.trace(name=trace_name):
                    input_data = {"function": trace_name, "args": _serialize_args(args), "kwargs": _serialize_args(kwargs)}
                    langfuse_context.update_current_trace(input=input_data)
                    result = await func(*args, **kwargs)
                    latency_ms = (time.time() - start_time) * 1000
                    output_data = {"result": _serialize_args(result), "latency_ms": latency_ms}
                    langfuse_context.update_current_trace(output=output_data)
                    logger.info(f"LLM call '{trace_name}' completed in {latency_ms:.2f}ms")
                    return result
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                logger.error(f"Error in traced call {trace_name} after {latency_ms:.2f}ms: {e}")
                langfuse_context.update_current_trace(output={"error": str(e), "latency_ms": latency_ms}, level="ERROR")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            trace_name = name or func.__name__
            start_time = time.time()
            if not LANGFUSE_AVAILABLE or not langfuse_context:
                return func(*args, **kwargs)
            try:
                with langfuse_context.trace(name=trace_name):
                    input_data = {"function": trace_name, "args": _serialize_args(args), "kwargs": _serialize_args(kwargs)}
                    langfuse_context.update_current_trace(input=input_data)
                    result = func(*args, **kwargs)
                    latency_ms = (time.time() - start_time) * 1000
                    output_data = {"result": _serialize_args(result), "latency_ms": latency_ms}
                    langfuse_context.update_current_trace(output=output_data)
                    logger.info(f"LLM call '{trace_name}' completed in {latency_ms:.2f}ms")
                    return result
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                logger.error(f"Error in traced call {trace_name} after {latency_ms:.2f}ms: {e}")
                langfuse_context.update_current_trace(output={"error": str(e), "latency_ms": latency_ms}, level="ERROR")
                raise

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@contextmanager
def trace_llm_operation(operation_name: str, input_data: Optional[dict] = None):
  
    start_time = time.time()
    if not LANGFUSE_AVAILABLE or not langfuse_context:
        yield
        return
    try:
        with langfuse_context.trace(name=operation_name):
            if input_data:
                langfuse_context.update_current_trace(input=input_data)
            logger.info(f"Starting LLM operation: {operation_name}")
            yield
            latency_ms = (time.time() - start_time) * 1000
            langfuse_context.update_current_trace(output={"status": "success", "latency_ms": latency_ms})
            logger.info(f"LLM operation '{operation_name}' completed successfully in {latency_ms:.2f}ms")
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"Error in LLM operation '{operation_name}' after {latency_ms:.2f}ms: {e}")
        langfuse_context.update_current_trace(output={"status": "error", "error": str(e), "latency_ms": latency_ms}, level="ERROR")
        raise


def trace_span(name: str):
  
    class SpanContext:
        def __enter__(self):
            if _langfuse_enabled and langfuse_context is not None:
                return langfuse_context.span(name=name)
            return None

        def __exit__(self, *args):
            pass

    return SpanContext()


def _serialize_args(obj: Any, max_depth: int = 2, current_depth: int = 0) -> Any:
   
    if current_depth >= max_depth:
        return f"<{type(obj).__name__}>"
    
    try:
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {
                k: _serialize_args(v, max_depth, current_depth + 1)
                for k, v in list(obj.items())[:10]  # Limit dict size
            }
        elif isinstance(obj, (list, tuple)):
            return [
                _serialize_args(item, max_depth, current_depth + 1)
                for item in obj[:5]  # Limit list size
            ]
        elif hasattr(obj, "__dict__"):
            # For custom objects, try to get string representation
            return str(obj)[:200]
        else:
            return str(obj)[:200]
    except Exception:
        return f"<{type(obj).__name__}>"


def flush():
   
    if _callback_handler and _langfuse_enabled:
        try:
            _callback_handler.flush()
        except Exception as e:
            logger.warning(f"Failed to flush Langfuse traces: {e}")


def get_callback_handler():
   
    global _callback_handler
    if not _langfuse_enabled:
        return None
    if _callback_handler is not None:
        return _callback_handler
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    try:
        from langfuse.callback import CallbackHandler
        _callback_handler = CallbackHandler(public_key=public_key, secret_key=secret_key, host=host)
        logger.info("Langfuse CallbackHandler created successfully")
        return _callback_handler
    except ImportError:
        pass
    try:
        from langfuse.langchain import CallbackHandler
        _callback_handler = CallbackHandler(public_key=public_key, secret_key=secret_key, host=host)
        logger.info("Langfuse CallbackHandler created successfully")
        return _callback_handler
    except Exception as e:
        logger.warning(f"Langfuse LangChain integration not available: {e}")
        return None


# Initialize Langfuse on module import
