from dataclasses import dataclass
import re

from app.schemas.document import ChunkingStrategy


@dataclass
class ChunkResult:
    chunk_index: int
    content: str
    page_number: int | None = None


class ChunkingService:
    def fixed_chunk(
        self,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        page_number: int | None = None,
        start_index: int = 0,
    ) -> list[ChunkResult]:
        cleaned_text = self._normalize_text(text)
        if not cleaned_text:
            return []

        chunks: list[ChunkResult] = []
        stride = chunk_size - chunk_overlap
        if stride <= 0:
            stride = chunk_size

        curr_index = start_index
        for start in range(0, len(cleaned_text), stride):
            end = min(start + chunk_size, len(cleaned_text))
            chunk_content = cleaned_text[start:end].strip()
            if chunk_content:
                chunks.append(
                    ChunkResult(
                        chunk_index=curr_index,
                        content=chunk_content,
                        page_number=page_number,
                    )
                )
                curr_index += 1
            if end >= len(cleaned_text):
                break

        return chunks

    def recursive_chunk(
        self,
        text: str,
        target_chunk_size: int = 500,
        chunk_overlap: int = 50,
        page_number: int | None = None,
        start_index: int = 0,
    ) -> list[ChunkResult]:
        cleaned_text = self._normalize_text(text)
        if not cleaned_text:
            return []

        splits = self._split_text_recursively(
            cleaned_text, target_chunk_size, chunk_overlap
        )
        chunks: list[ChunkResult] = []
        curr_index = start_index

        for content in splits:
            trimmed = content.strip()
            if trimmed:
                chunks.append(
                    ChunkResult(
                        chunk_index=curr_index,
                        content=trimmed,
                        page_number=page_number,
                    )
                )
                curr_index += 1

        return chunks

    def chunk_document(
        self,
        text_or_pages: str | list[tuple[int, str]],
        strategy: ChunkingStrategy,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> list[ChunkResult]:
        all_chunks: list[ChunkResult] = []
        curr_index = 0

        if isinstance(text_or_pages, str):
            pages = [(None, text_or_pages)]
        else:
            pages = text_or_pages

        for page_num, text in pages:
            if not text.strip():
                continue
            if strategy == ChunkingStrategy.FIXED:
                page_chunks = self.fixed_chunk(
                    text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    page_number=page_num,
                    start_index=curr_index,
                )
            else:
                page_chunks = self.recursive_chunk(
                    text,
                    target_chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    page_number=page_num,
                    start_index=curr_index,
                )
            all_chunks.extend(page_chunks)
            curr_index += len(page_chunks)

        return all_chunks

    def _split_text_recursively(
        self, text: str, target_size: int, overlap: int
    ) -> list[str]:
        separators = ["\n\n", "\n", ". ", " ", ""]

        def _split(txt: str, seps: list[str]) -> list[str]:
            if len(txt) <= target_size:
                return [txt] if txt.strip() else []

            if not seps:
                # Fallback to hard character split
                res = []
                stride = target_size - overlap if target_size > overlap else target_size
                for i in range(0, len(txt), stride):
                    part = txt[i : i + target_size]
                    if part.strip():
                        res.append(part)
                return res

            sep = seps[0]
            next_seps = seps[1:]

            if sep == "":
                splits = list(txt)
            else:
                splits = txt.split(sep)

            docs: list[str] = []
            current_doc: list[str] = []
            total_len = 0

            for s in splits:
                s_len = len(s) + (len(sep) if sep != "" else 0)
                if total_len + s_len > target_size and current_doc:
                    joined = (sep if sep != "" else "").join(current_doc)
                    docs.append(joined)

                    # Overlap handling: retain tail elements from current_doc
                    while current_doc and total_len > overlap:
                        removed = current_doc.pop(0)
                        total_len -= len(removed) + (len(sep) if sep != "" else 0)

                current_doc.append(s)
                total_len += s_len

            if current_doc:
                joined = (sep if sep != "" else "").join(current_doc)
                docs.append(joined)

            final_result: list[str] = []
            for d in docs:
                if len(d) > target_size:
                    final_result.extend(_split(d, next_seps))
                elif d.strip():
                    final_result.append(d)

            return final_result

        return _split(text, separators)

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        # Collapse whitespace and strip control characters
        normalized = re.sub(r"\s+", " ", text)
        return normalized.strip()


chunking_service = ChunkingService()
