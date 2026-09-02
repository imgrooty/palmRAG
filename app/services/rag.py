from app.integrations.groq import groq_service
from app.schemas.chat import Source
from app.services.retrieval import RetrievedChunk, retrieval_service


class RAGService:
    async def process_rag(
        self, message: str, history: list[dict[str, str]]
    ) -> tuple[str, list[Source]]:
        # 1. Step 1: Rewrite query with conversation context
        standalone_query = await groq_service.rewrite_query(message, history)

        # 2. Step 2: Retrieve relevant chunks from vector store
        chunks: list[RetrievedChunk] = await retrieval_service.retrieve_relevant_chunks(
            query=standalone_query,
            top_k=5,
            score_threshold=0.35,
        )

        # FR-19: If retrieval returns no relevant chunks above threshold, return explicit message
        if not chunks:
            return (
                "I couldn't find anything relevant to that in the uploaded documents.",
                [],
            )

        # 3. Step 3: Build explicit RAG prompt manually (No LangChain abstractions)
        prompt = self.build_rag_prompt(standalone_query, history, chunks)

        # 4. Generate answer via Groq LLM
        answer = await groq_service.generate_rag_answer(prompt)

        # Format deduplicated sources
        sources: list[Source] = []
        seen = set()
        for chunk in chunks:
            key = (chunk.filename, chunk.page_number)
            if key not in seen:
                seen.add(key)
                sources.append(Source(document=chunk.filename, page=chunk.page_number))

        return answer, sources

    def build_rag_prompt(
        self, query: str, history: list[dict[str, str]], context: list[RetrievedChunk]
    ) -> str:
        context_blocks = []
        for c in context:
            page_info = f", page {c.page_number}" if c.page_number is not None else ""
            context_blocks.append(f"[Source: {c.filename}{page_info}]\n{c.content}")
        context_str = "\n\n".join(context_blocks)

        history_str = (
            "\n".join(f"{t['role']}: {t['content']}" for t in history[-6:])
            if history
            else "None"
        )

        return (
            "You are answering questions using only the provided document context.\n"
            "If the answer cannot be determined from the context, state explicitly that you could not find relevant information in the documents.\n\n"
            f"Conversation History:\n{history_str}\n\n"
            f"Retrieved Document Context:\n{context_str}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )


rag_service = RAGService()
