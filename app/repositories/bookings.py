"""Booking repository — Motor (MongoDB) implementation.

Public method signatures are identical to the old SQLAlchemy version.
"""

import uuid
from datetime import date, datetime, time, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase


class BookingRepository:
    async def create_booking(
        self,
        db: AsyncIOMotorDatabase,
        session_id: str,
        name: str,
        email: str,
        interview_date: date,
        interview_time: time,
        status: str = "confirmed",
    ) -> dict[str, Any]:
        """Persist a confirmed interview booking.

        Args:
            db: Motor database instance.
            session_id: Redis conversation session ID.
            name: Interviewee full name.
            email: Validated email address.
            interview_date: Scheduled date.
            interview_time: Scheduled time.
            status: Booking status, defaults to 'confirmed'.

        Returns:
            Inserted booking dict (includes '_id').
        """
        booking: dict[str, Any] = {
            "_id": uuid.uuid4(),
            "session_id": session_id,
            "name": name,
            "email": email,
            # Store as ISO strings so they survive BSON round-trips cleanly.
            "interview_date": interview_date.isoformat(),
            "interview_time": interview_time.isoformat(),
            "status": status,
            "created_at": datetime.now(tz=timezone.utc),
        }
        await db["bookings"].insert_one(booking)
        return booking

    async def get_booking_by_id(
        self, db: AsyncIOMotorDatabase, booking_id: uuid.UUID
    ) -> dict[str, Any] | None:
        """Fetch a booking by its _id.

        Args:
            db: Motor database instance.
            booking_id: Booking _id.

        Returns:
            Booking dict or None.
        """
        return await db["bookings"].find_one({"_id": booking_id})

    async def get_bookings_by_session_id(
        self, db: AsyncIOMotorDatabase, session_id: str
    ) -> list[dict[str, Any]]:
        """Retrieve all bookings for a conversation session, newest first.

        Args:
            db: Motor database instance.
            session_id: Redis conversation session ID.

        Returns:
            List of booking dicts ordered by created_at descending.
        """
        cursor = db["bookings"].find(
            {"session_id": session_id},
            sort=[("created_at", -1)],
        )
        return await cursor.to_list(length=None)


booking_repository = BookingRepository()
