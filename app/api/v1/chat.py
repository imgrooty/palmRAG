from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.db.database import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.booking import booking_service
from app.services.chat_memory import chat_memory_service
from app.services.rag import rag_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_turn(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    try:
        session_id = request.session_id
        message = request.message

        # 1. Load prior conversation history from Redis
        history = await chat_memory_service.get_history(session_id)

        # 2. Check for interview booking processing first
        booking_obj, booking_response_text = await booking_service.process_booking_turn(
            db_session=db,
            session_id=session_id,
            message=message,
            history=history,
        )

        if booking_obj is not None or booking_response_text is not None:
            final_answer = booking_response_text or "Booking processed successfully."
            await chat_memory_service.add_turn(
                session_id=session_id,
                user_message=message,
                assistant_answer=final_answer,
            )
            return ChatResponse(
                answer=final_answer,
                sources=[],
                booking=booking_obj,
            )

        # 3. Perform standard RAG pipeline
        answer, sources = await rag_service.process_rag(
            message=message,
            history=history,
        )

        # 4. Save turn back to Redis
        await chat_memory_service.add_turn(
            session_id=session_id,
            user_message=message,
            assistant_answer=answer,
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            booking=None,
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint for session {request.session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {e}",
        ) from e
