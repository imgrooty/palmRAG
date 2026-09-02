from app.schemas.document import ChunkingStrategy
from app.services.chunking import chunking_service


def test_fixed_chunking_basic():
    text = "Hello world! This is a test document to verify fixed chunking behavior."
    chunks = chunking_service.fixed_chunk(text, chunk_size=25, chunk_overlap=5)

    assert len(chunks) > 0
    assert chunks[0].chunk_index == 0
    assert len(chunks[0].content) <= 25


def test_recursive_chunking_respects_paragraphs():
    text = (
        "Paragraph one is about Python backend engineering.\n\n"
        "Paragraph two focuses on retrieval augmented generation and vector databases."
    )
    chunks = chunking_service.recursive_chunk(
        text, target_chunk_size=70, chunk_overlap=10
    )

    assert len(chunks) >= 2
    assert "Paragraph one" in chunks[0].content
    assert "Paragraph two" in chunks[1].content


def test_chunking_comparison_fixed_vs_recursive():
    sample_text = (
        "FastAPI is a modern, fast web framework for building APIs with Python 3.11+. "
        "It uses standard Python type hints to deliver high performance and automatic Swagger docs.\n\n"
        "Qdrant is an open-source vector similarity search engine. It provides production-ready "
        "vector database capabilities with low latency and high scalability."
    )

    fixed_chunks = chunking_service.chunk_document(
        sample_text, strategy=ChunkingStrategy.FIXED, chunk_size=100, chunk_overlap=10
    )
    recursive_chunks = chunking_service.chunk_document(
        sample_text,
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(fixed_chunks) > 0
    assert len(recursive_chunks) > 0
    # Fixed cuts strictly at character limits, while recursive respects paragraph/sentence boundaries
    assert (
        recursive_chunks[0].content.endswith(".")
        or "\n" not in recursive_chunks[0].content
    )


def test_chunking_empty_and_whitespace_input():
    empty_chunks = chunking_service.chunk_document("", strategy=ChunkingStrategy.FIXED)
    assert empty_chunks == []

    whitespace_chunks = chunking_service.chunk_document(
        "   \n\n  ", strategy=ChunkingStrategy.RECURSIVE
    )
    assert whitespace_chunks == []


def test_chunking_multipage_pdf():
    pages = [
        (1, "Page 1 content about applicant's work history."),
        (2, "Page 2 content details applicant's education and degrees."),
    ]
    chunks = chunking_service.chunk_document(
        pages, strategy=ChunkingStrategy.FIXED, chunk_size=50
    )

    assert len(chunks) >= 2
    page1_chunks = [c for c in chunks if c.page_number == 1]
    page2_chunks = [c for c in chunks if c.page_number == 2]

    assert len(page1_chunks) > 0
    assert len(page2_chunks) > 0
    assert "history" in page1_chunks[0].content
    assert "education" in page2_chunks[0].content
