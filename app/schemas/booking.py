from datetime import date, time
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class InterviewBooking(BaseModel):
    id: UUID | None = None
    name: str
    email: EmailStr
    date: date
    time: time
    status: str = "confirmed"

    model_config = ConfigDict(from_attributes=True)


class PartialBookingState(BaseModel):
    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None


class ExtractedBookingInfo(BaseModel):
    has_booking_intent: bool = Field(
        default=False,
        description="True if user expressed intent to book an interview/meeting",
    )
    name: str | None = None
    email: str | None = None
    date: str | None = Field(
        default=None, description="ISO format date string YYYY-MM-DD if extracted"
    )
    time: str | None = Field(
        default=None, description="24-hour time string HH:MM or HH:MM:SS if extracted"
    )
