from unittest.mock import AsyncMock, patch

import pytest

from app.schemas.booking import ExtractedBookingInfo
from app.services.booking import booking_service


@pytest.mark.asyncio
async def test_full_booking_flow_single_turn(async_session):
    extracted_info = ExtractedBookingInfo(
        has_booking_intent=True,
        name="Bikram Sharma",
        email="bikram@example.com",
        date="2026-09-15",
        time="15:00:00",
    )

    with (
        patch(
            "app.services.booking.redis_service.get_partial_booking",
            AsyncMock(return_value={}),
        ),
        patch(
            "app.services.booking.groq_service.extract_booking_info",
            AsyncMock(return_value=extracted_info),
        ),
        patch(
            "app.services.booking.redis_service.clear_partial_booking", AsyncMock()
        ) as mock_clear,
    ):
        booking, message = await booking_service.process_booking_turn(
            db_session=async_session,
            session_id="session123",
            message="Book an interview for Bikram, email bikram@example.com on Sep 15 2026 at 3pm",
            history=[],
        )

        assert booking is not None
        assert booking.name == "Bikram Sharma"
        assert booking.email == "bikram@example.com"
        assert str(booking.date) == "2026-09-15"
        assert str(booking.time) == "15:00:00"
        assert "booked your interview" in message
        mock_clear.assert_called_once_with("session123")


@pytest.mark.asyncio
async def test_partial_booking_flow_missing_email(async_session):
    extracted_info = ExtractedBookingInfo(
        has_booking_intent=True,
        name="Bikram Sharma",
        email=None,
        date="2026-09-15",
        time="15:00:00",
    )

    with (
        patch(
            "app.services.booking.redis_service.get_partial_booking",
            AsyncMock(return_value={}),
        ),
        patch(
            "app.services.booking.groq_service.extract_booking_info",
            AsyncMock(return_value=extracted_info),
        ),
        patch(
            "app.services.booking.redis_service.save_partial_booking", AsyncMock()
        ) as mock_save,
    ):
        booking, follow_up = await booking_service.process_booking_turn(
            db_session=async_session,
            session_id="session123",
            message="Book an interview for Bikram on Sep 15 at 3pm",
            history=[],
        )

        assert booking is None
        assert "email address" in follow_up
        mock_save.assert_called_once()


def test_date_and_time_parsing_utility():
    assert str(booking_service._parse_date("2026-09-15")) == "2026-09-15"
    assert str(booking_service._parse_date("September 15, 2026")) == "2026-09-15"
    assert booking_service._parse_date("invalid-date") is None

    assert str(booking_service._parse_time("15:00:00")) == "15:00:00"
    assert str(booking_service._parse_time("3:00 PM")) == "15:00:00"
    assert str(booking_service._parse_time("3pm")) == "15:00:00"
    assert booking_service._parse_time("invalid-time") is None
