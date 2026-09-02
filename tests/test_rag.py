from unittest.mock import AsyncMock, patch

import pytest

from app.services.rag import rag_service
from app.services.retrieval import RetrievedChunk


@pytest.mark.asyncio
async def test_rag_pipeline_with_retrieved_chunks():
    history = [{"role": "user", "content": "Tell me about John"}]
    query = "What framework does he use?"

    retrieved = [
        RetrievedChunk(
            content="John builds REST APIs using FastAPI and PostgreSQL.",
            filename="john_resume.pdf",
            page_number=2,
            score=0.85,
        )
    ]

    with (
        patch(
            "app.services.rag.groq_service.rewrite_query",
            AsyncMock(return_value="What framework does John use?"),
        ),
        patch(
            "app.services.rag.retrieval_service.retrieve_relevant_chunks",
            AsyncMock(return_value=retrieved),
        ),
        patch(
            "app.services.rag.groq_service.generate_rag_answer",
            AsyncMock(return_value="John uses FastAPI for REST APIs."),
        ),
    ):
        answer, sources = await rag_service.process_rag(query, history)

        assert "FastAPI" in answer
        assert len(sources) == 1
        assert sources[0].document == "john_resume.pdf"
        assert sources[0].page == 2


@pytest.mark.asyncio
async def test_rag_pipeline_no_relevant_chunks():
    with (
        patch(
            "app.services.rag.groq_service.rewrite_query",
            AsyncMock(return_value="Unrelated query"),
        ),
        patch(
            "app.services.rag.retrieval_service.retrieve_relevant_chunks",
            AsyncMock(return_value=[]),
        ),
    ):
        answer, sources = await rag_service.process_rag(
            "What is the distance to Mars?", []
        )

        assert (
            answer
            == "I couldn't find anything relevant to that in the uploaded documents."
        )
        assert sources == []


def test_build_rag_prompt_formatting():
    chunks = [
        RetrievedChunk(
            content="FastAPI documentation snippet.", filename="docs.pdf", page_number=1
        ),
    ]
    prompt = rag_service.build_rag_prompt("How to use FastAPI?", [], chunks)

    assert "Retrieved Document Context:" in prompt
    assert "[Source: docs.pdf, page 1]" in prompt
    assert "FastAPI documentation snippet." in prompt
    assert "Question: How to use FastAPI?" in prompt
