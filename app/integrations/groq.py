import json
from datetime import datetime
from pathlib import Path
from typing import Any

from groq import AsyncGroq

from app.core.config import settings
from app.core.logging import logger
from app.schemas.booking import ExtractedBookingInfo


class GroqService:
    def __init__(self) -> None:
        self._client: AsyncGroq | None = None

    def get_client(self) -> AsyncGroq:
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured in environment variables or .env file."
            )
        if self._client is None:
            self._client = AsyncGroq(api_key=settings.groq_api_key)
        return self._client

    async def rewrite_query(self, query: str, history: list[dict[str, str]]) -> str:
        if not history:
            return query

        client = self.get_client()
        history_text = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in history[-6:]
        )
        prompt = (
            "Given the following conversation history and a follow-up question, "
            "rephrase the follow-up question to be a standalone search query that preserves all context. "
            "Do NOT answer the question, only output the rewritten question.\n\n"
            f"History:\n{history_text}\n\n"
            f"Follow-up Question: {query}\n"
            "Standalone Question:"
        )

        try:
            response = await client.chat.completions.create(
                model=settings.groq_fast_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a query rewriting assistant. Output only the rewritten query.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            rewritten = response.choices[0].message.content or query
            return rewritten.strip()
        except Exception as e:
            logger.error(f"Failed to rewrite query via Groq: {e}")
            return query

    async def generate_rag_answer(self, prompt: str) -> str:
        client = self.get_client()
        try:
            response = await client.chat.completions.create(
                model=settings.groq_chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that answers questions accurately based on provided document context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1000,
            )
            answer = response.choices[0].message.content or ""
            return answer.strip()
        except Exception as e:
            logger.error(f"Failed to generate RAG answer via Groq: {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e

    async def extract_booking_info(
        self,
        user_message: str,
        partial_state: dict[str, Any],
        history: list[dict[str, str]],
    ) -> ExtractedBookingInfo:
        client = self.get_client()

        system_instruction = (
            "You are an interview booking extraction assistant. Your task is to inspect the user's message, "
            "conversation history, and current partial booking state to extract booking details.\n"
            f"The current date is {datetime.now().strftime('%Y-%m-%d')}. Resolve relative dates (like 'tomorrow', 'next week') to absolute dates.\n"
            "Respond ONLY with a valid JSON object with the following fields:\n"
            "{\n"
            '  "has_booking_intent": bool (true if user expresses intent to schedule/book an interview/meeting),\n'
            '  "name": string or null,\n'
            '  "email": string or null,\n'
            '  "date": string or null (in YYYY-MM-DD format if possible or raw extracted text),\n'
            '  "time": string or null (in 24h format HH:MM:SS or HH:MM if possible)\n'
            "}\n"
            "Merge any existing partial state unless updated by the user message."
        )

        user_content = (
            f"Current Partial State: {json.dumps(partial_state)}\n"
            f"User Message: {user_message}\n"
        )

        try:
            response = await client.chat.completions.create(
                model=settings.groq_fast_model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=300,
            )
            raw_json = response.choices[0].message.content or "{}"
            parsed = json.loads(raw_json)
            return ExtractedBookingInfo(**parsed)
        except Exception as e:
            logger.error(f"Failed to extract booking info via Groq: {e}")
            return ExtractedBookingInfo(has_booking_intent=False)

    async def transcribe_audio(self, audio_file_path: str | Path) -> str:
        client = self.get_client()
        file_path = Path(audio_file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            with open(file_path, "rb") as audio_file:
                transcription = await client.audio.transcriptions.create(
                    file=(file_path.name, audio_file),
                    model=settings.groq_whisper_model,
                    response_format="text",
                )
            return str(transcription).strip()
        except Exception as e:
            logger.error(f"Failed to transcribe audio via Groq Whisper: {e}")
            raise RuntimeError(f"Transcription failed: {e}") from e


groq_service = GroqService()
