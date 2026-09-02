from app.integrations.redis import redis_service


class ChatMemoryService:
    async def get_history(self, session_id: str) -> list[dict[str, str]]:
        return await redis_service.get_history(session_id)

    async def add_turn(
        self, session_id: str, user_message: str, assistant_answer: str
    ) -> None:
        await redis_service.append_turn(session_id, user_message, assistant_answer)


chat_memory_service = ChatMemoryService()
