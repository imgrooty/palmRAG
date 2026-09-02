import asyncio
import os
from pathlib import Path
import tempfile

from app.core.logging import logger
from app.integrations.groq import groq_service


class TranscriptionService:
    async def extract_audio(self, video_path: str | Path) -> Path:
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found at {video_file}")

        temp_audio_fd, temp_audio_path = tempfile.mkstemp(suffix=".mp3")
        os.close(temp_audio_fd)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_file),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-ar",
            "16000",
            "-ac",
            "1",
            temp_audio_path,
        ]

        logger.info(f"Running ffmpeg to extract audio from {video_file}...")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            err_msg = stderr.decode() if stderr else "Unknown ffmpeg error"
            logger.error(f"ffmpeg extraction failed: {err_msg}")
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            raise RuntimeError(f"ffmpeg audio extraction failed: {err_msg}")

        logger.info(f"Audio extracted successfully to {temp_audio_path}")
        return Path(temp_audio_path)

    async def transcribe_video(self, video_path: str | Path) -> str:
        audio_path = await self.extract_audio(video_path)
        try:
            logger.info(
                "Sending extracted audio to Groq Whisper API for transcription..."
            )
            transcript = await groq_service.transcribe_audio(audio_path)
            logger.info("Video transcription completed successfully.")
            return transcript
        finally:
            if audio_path.exists():
                try:
                    audio_path.unlink()
                except FileNotFoundError:
                    pass


transcription_service = TranscriptionService()
