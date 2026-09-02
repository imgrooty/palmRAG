from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.services.transcription import transcription_service


@pytest.mark.asyncio
async def test_transcribe_video():
    dummy_video = Path("dummy_video.mp4")

    with (
        patch.object(Path, "exists", return_value=True),
        patch(
            "app.services.transcription.transcription_service.extract_audio",
            AsyncMock(return_value=Path("temp_audio.mp3")),
        ),
        patch(
            "app.services.transcription.groq_service.transcribe_audio",
            AsyncMock(return_value="Hello this is a video intro."),
        ),
    ):
        transcript = await transcription_service.transcribe_video(dummy_video)
        assert transcript == "Hello this is a video intro."
