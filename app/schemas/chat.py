from pydantic import BaseModel, Field

from app.schemas.booking import InterviewBooking


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=255)
    message: str = Field(..., min_length=1, max_length=4000)


class Source(BaseModel):
    document: str
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    booking: InterviewBooking | None = None
