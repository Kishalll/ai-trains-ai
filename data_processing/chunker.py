import re
from typing import Any


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    # Split by paragraphs first if available
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > chunk_size:
            # Paragraph is too large, split by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                    current_chunk = f"{current_chunk} {sentence}".strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    if len(sentence) > chunk_size:
                        # Slice long single sentence into pieces with overlap
                        step = max(1, chunk_size - overlap)
                        for i in range(0, len(sentence), step):
                            piece = sentence[i : i + chunk_size]
                            if piece:
                                chunks.append(piece)
                        current_chunk = ""
                    else:
                        current_chunk = sentence
        else:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_documents(
    docs: list[dict[str, Any]], chunk_size: int = 500, overlap: int = 50
) -> list[dict[str, Any]]:
    chunked_docs = []
    source_counters: dict[str, int] = {}

    for doc in docs:
        raw_text = doc.get("text", "")
        source = doc.get("source", "unknown")
        metadata = doc.get("metadata", {}).copy()

        text_pieces = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
        for piece in text_pieces:
            current_idx = source_counters.get(source, 0)
            source_counters[source] = current_idx + 1

            chunk_meta = metadata.copy()
            chunk_meta["chunk_index"] = current_idx
            chunk_meta["source"] = source

            chunked_docs.append(
                {
                    "text": piece,
                    "source": source,
                    "metadata": chunk_meta,
                    "id": f"{source}_chunk_{current_idx}",
                }
            )

    return chunked_docs
