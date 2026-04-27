import hashlib
import json
import logging

import redis

from config import settings

logger = logging.getLogger(__name__)

_redis_client: redis.Redis = None
_QA_PREFIX = "qa:"
_DEFAULT_TTL = 3600


def init_redis() -> None:
    global _redis_client
    _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    _redis_client.ping()
    logger.info("Redis connected successfully")


def close_redis() -> None:
    global _redis_client
    if _redis_client:
        _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed")


def _question_key(question: str) -> str:
    h = hashlib.sha256(question.lower().strip().encode()).hexdigest()
    return f"{_QA_PREFIX}{h}"


def get_cached_answer(question: str) -> dict | None:
    if not _redis_client:
        return None
    try:
        raw = _redis_client.get(_question_key(question))
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning(f"Redis get failed: {e}")
        return None


def set_cached_answer(question: str, answer: dict) -> None:
    if not _redis_client:
        return
    try:
        _redis_client.setex(_question_key(question), _DEFAULT_TTL, json.dumps(answer))
    except Exception as e:
        logger.warning(f"Redis set failed: {e}")


def clear_qa_cache() -> int:
    if not _redis_client:
        return 0
    try:
        keys = _redis_client.keys(f"{_QA_PREFIX}*")
        if keys:
            deleted = _redis_client.delete(*keys)
            logger.info(f"Redis cache cleared: {deleted} Q&A entries removed")
            return deleted
        return 0
    except Exception as e:
        logger.warning(f"Redis cache clear failed: {e}")
        return 0
