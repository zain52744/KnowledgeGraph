import hashlib
import json
import logging

import redis

from config import settings

logger = logging.getLogger(__name__)

_QA_PREFIX = "qa:"


class RedisCache:

    def __init__(self, url: str, ttl: int = 3600) -> None:
        self._url = url
        self._ttl = ttl
        self._client: redis.Redis = None

    def connect(self) -> None:
        self._client = redis.from_url(self._url, decode_responses=True)
        self._client.ping()
        logger.info("Redis connected successfully")

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Redis connection closed")

    def get(self, question: str) -> dict | None:
        if not self._client:
            return None
        try:
            raw = self._client.get(self._key(question))
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None

    def set(self, question: str, answer: dict) -> None:
        if not self._client:
            return
        try:
            self._client.setex(self._key(question), self._ttl, json.dumps(answer))
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")

    def clear(self) -> int:
        if not self._client:
            return 0
        try:
            keys = self._client.keys(f"{_QA_PREFIX}*")
            if keys:
                deleted = self._client.delete(*keys)
                logger.info(f"Redis cache cleared: {deleted} Q&A entries removed")
                return deleted
            return 0
        except Exception as e:
            logger.warning(f"Redis cache clear failed: {e}")
            return 0

    def _key(self, question: str) -> str:
        h = hashlib.sha256(question.lower().strip().encode()).hexdigest()
        return f"{_QA_PREFIX}{h}"


cache = RedisCache(url=settings.redis_url)
