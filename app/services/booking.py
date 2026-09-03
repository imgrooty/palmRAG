from datetime import date, datetime, time
import re
from typing import Any

from pydantic import ValidationError
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.logging import logger
from app.integrations.groq import groq_service
from app.integrations.redis import redis_service
from app.repositories.bookings import booking_repository
from app.schemas.booking import ExtractedBookingInfo, InterviewBooking


class BookingService:
    async def process_booking_turn(
        self,
        db_session: AsyncIOMotorDatabase,
        session_id: str,
        message: str,
        history: list[dict[str, str]],
    ) -> tuple[InterviewBooking | None, str | None]:
        # 1. Fetch partial state from Redis
        partial_state = await redis_service.get_partial_booking(session_id)

        # 2. Extract booking intent and fields using Groq
        extracted: ExtractedBookingInfo = await groq_service.extract_booking_info(
            user_message=message,
            partial_state=partial_state,
            history=history,
        )

        if not extracted.has_booking_intent and not partial_state:
            return None, None

        # 3. Merge extracted fields into partial state
        merged_state = dict(partial_state)
        if extracted.name:
            merged_state["name"] = extracted.name
        if extracted.email:
            merged_state["email"] = extracted.email
        if extracted.date:
            merged_state["date"] = extracted.date
        if extracted.time:
            merged_state["time"] = extracted.time

        # 4. Attempt to parse and validate against Pydantic InterviewBooking model
        try_booking, missing_fields = self._try_build_booking(merged_state)

        if try_booking is not None:
            # Full validation success! Persist to DB and clear Redis partial state
            db_booking = await booking_repository.create_booking(
                db=db_session,
                session_id=session_id,
                name=try_booking.name,
                email=try_booking.email,
                interview_date=try_booking.date,
                interview_time=try_booking.time,
                status="confirmed",
            )
            await redis_service.clear_partial_booking(session_id)

            confirmed_booking = InterviewBooking(
                id=db_booking["_id"],
                name=db_booking["name"],
                email=db_booking["email"],  # type: ignore[arg-type]
                date=try_booking.date,
                time=try_booking.time,
                status=db_booking.get("status", "confirmed"),
            )

            date_str = confirmed_booking.date.strftime("%B %d, %Y")
            time_str = confirmed_booking.time.strftime("%I:%M %p")
            confirmation_message = (
                f"You're all set! I've booked your interview for {date_str} at {time_str}. "
                f"A confirmation has been recorded for {confirmed_booking.name} ({confirmed_booking.email})."
            )
            return confirmed_booking, confirmation_message

        # Missing or invalid fields - update partial state in Redis and prompt user
        await redis_service.save_partial_booking(session_id, merged_state)

        follow_up_prompt = self._build_followup_question(missing_fields, merged_state)
        return None, follow_up_prompt

    def _try_build_booking(
        self, state: dict[str, Any]
    ) -> tuple[InterviewBooking | None, list[str]]:
        missing: list[str] = []

        name = state.get("name")
        email = state.get("email")
        raw_date = state.get("date")
        raw_time = state.get("time")

        if not name:
            missing.append("name")

        if not email:
            missing.append("email")

        parsed_date = self._parse_date(raw_date) if raw_date else None
        if parsed_date is None:
            missing.append("date")

        parsed_time = self._parse_time(raw_time) if raw_time else None
        if parsed_time is None:
            missing.append("time")

        if missing:
            return None, missing

        try:
            booking = InterviewBooking(
                name=name,
                email=email,
                date=parsed_date,  # type: ignore[arg-type]
                time=parsed_time,  # type: ignore[arg-type]
                status="confirmed",
            )
            return booking, []
        except ValidationError as ve:
            logger.warning(f"Booking validation failed: {ve}")
            # Identify which fields failed validation
            for error in ve.errors():
                loc = str(error.get("loc", ()))
                if "email" in loc and "email" not in missing:
                    missing.append("email")
                if "date" in loc and "date" not in missing:
                    missing.append("date")
                if "time" in loc and "time" not in missing:
                    missing.append("time")
            return None, missing

    def _parse_date(self, val: Any) -> date | None:
        if isinstance(val, date):
            return val
        if not isinstance(val, str):
            return None

        val_str = val.strip()
        # ISO format YYYY-MM-DD
        iso_match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", val_str)
        if iso_match:
            try:
                return date(
                    int(iso_match.group(1)),
                    int(iso_match.group(2)),
                    int(iso_match.group(3)),
                )
            except ValueError:
                return None

        # Common format MM/DD/YYYY or DD/MM/YYYY
        for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(val_str, fmt).date()
            except ValueError:
                pass

        return None

    def _parse_time(self, val: Any) -> time | None:
        if isinstance(val, time):
            return val
        if not isinstance(val, str):
            return None

        val_str = val.strip().lower()

        # 12-hour or 24-hour formats
        for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M%p", "%I %p", "%I%p"):
            try:
                return datetime.strptime(val_str, fmt).time()
            except ValueError:
                pass

        # Regex for 3pm, 3:30pm, 15:00
        time_match = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", val_str)
        if time_match:
            hr = int(time_match.group(1))
            mn = int(time_match.group(2)) if time_match.group(2) else 0
            meridiem = time_match.group(3)

            if meridiem:
                if meridiem == "pm" and hr < 12:
                    hr += 12
                elif meridiem == "am" and hr == 12:
                    hr = 0

            if 0 <= hr <= 23 and 0 <= mn <= 59:
                return time(hr, mn)

        return None

    def _build_followup_question(
        self, missing_fields: list[str], current_state: dict[str, Any]
    ) -> str:
        prompt_parts = []
        if "name" in missing_fields:
            prompt_parts.append("your full name")
        if "email" in missing_fields:
            prompt_parts.append("your email address")
        if "date" in missing_fields:
            prompt_parts.append("the preferred date for the interview")
        if "time" in missing_fields:
            prompt_parts.append("the preferred time for the interview")

        if len(prompt_parts) == 1:
            req_str = prompt_parts[0]
        elif len(prompt_parts) == 2:
            req_str = f"{prompt_parts[0]} and {prompt_parts[1]}"
        else:
            req_str = f"{', '.join(prompt_parts[:-1])}, and {prompt_parts[-1]}"

        return (
            f"To complete your interview booking, could you please provide {req_str}?"
        )


booking_service = BookingService()
