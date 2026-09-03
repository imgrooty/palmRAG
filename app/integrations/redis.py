import orjson
from typing import Any

from redis.asyncio import Redis

from app.core.config import settings
from app.core.logging import logger


class RedisService:
    def __init__(self) -> None:
        self.redis_url = settings.redis_url
        self._client: Redis | None = None

    def get_client(self) -> Redis:
        if self._client is None:
            self._client = Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    async def get_history(self, session_id: str) -> list[dict[str, str]]:
        client = self.get_client()
        key = f"conversation:{session_id}"
        data = await client.get(key)
        if not data:
            return []
        try:
            return orjson.loads(data)
        except Exception as e:
            logger.error(
                f"Error parsing conversation history for session {session_id}: {e}"
            )
            return []

    async def append_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_answer: str,
        ttl: int = 86400,
    ) -> None:
        """Persist a completed conversation turn.

        Uses a pipeline to batch the GET + SET into a single network round-trip.
        """
        client = self.get_client()
        key = f"conversation:{session_id}"

        async with client.pipeline(transaction=False) as pipe:
            pipe.get(key)
            (existing_raw,) = await pipe.execute()

        history: list[dict[str, str]] = []
        if existing_raw:
            try:
                history = orjson.loads(existing_raw)
            except Exception as e:
                logger.error(
                    "Error parsing conversation history for session %s: %s",
                    session_id,
                    e,
                )

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_answer})

        # Keep last 20 turns max (40 messages) to avoid ballooning memory.
        if len(history) > 40:
            history = history[-40:]

        await client.set(key, orjson.dumps(history).decode(), ex=ttl)

    async def get_partial_booking(self, session_id: str) -> dict[str, Any]:
        client = self.get_client()
        key = f"booking:{session_id}"
        data = await client.get(key)
        if not data:
            return {}
        try:
            return orjson.loads(data)
        except Exception as e:
            logger.error(f"Error parsing partial booking for session {session_id}: {e}")
            return {}

    async def save_partial_booking(
        self, session_id: str, partial_data: dict[str, Any], ttl: int = 86400
    ) -> None:
        client = self.get_client()
        key = f"booking:{session_id}"
        await client.set(key, orjson.dumps(partial_data).decode(), ex=ttl)

    async def clear_partial_booking(self, session_id: str) -> None:
        client = self.get_client()
        key = f"booking:{session_id}"
        await client.delete(key)

    async def is_healthy(self) -> bool:
        try:
            client = self.get_client()
            await client.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


redis_service = RedisService()
