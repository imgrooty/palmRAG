import uuid
from datetime import date, time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.booking import BookingModel


class BookingRepository:
    async def create_booking(
        self,
        session: AsyncSession,
        session_id: str,
        name: str,
        email: str,
        interview_date: date,
        interview_time: time,
        status: str = "confirmed",
    ) -> BookingModel:
        booking = BookingModel(
            id=uuid.uuid4(),
            session_id=session_id,
            name=name,
            email=email,
            interview_date=interview_date,
            interview_time=interview_time,
            status=status,
        )
        session.add(booking)
        await session.flush()
        return booking

    async def get_booking_by_id(
        self, session: AsyncSession, booking_id: uuid.UUID
    ) -> BookingModel | None:
        result = await session.execute(
            select(BookingModel).where(BookingModel.id == booking_id)
        )
        return result.scalar_one_or_none()

    async def get_bookings_by_session_id(
        self, session: AsyncSession, session_id: str
    ) -> list[BookingModel]:
        result = await session.execute(
            select(BookingModel)
            .where(BookingModel.session_id == session_id)
            .order_by(BookingModel.created_at.desc())
        )
        return list(result.scalars().all())


booking_repository = BookingRepository()
