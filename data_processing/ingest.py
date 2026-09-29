import csv
import json
from pathlib import Path
from typing import Any


def read_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def read_csv_file(path: Path) -> list[str]:
    chunks = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            items = [f"{k}: {v}" for k, v in row.items() if v and v.strip()]
            if items:
                row_text = f"Row {i} | " + " | ".join(items)
                chunks.append(row_text)
    return chunks


def read_json_file(path: Path) -> list[str]:
    chunks = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    if isinstance(data, list):
        for i, item in enumerate(data, start=1):
            if isinstance(item, dict):
                formatted = [f"{k}: {v}" for k, v in item.items() if v is not None]
                chunks.append(f"Item {i} | " + " | ".join(formatted))
            else:
                chunks.append(str(item))
    elif isinstance(data, dict):
        for k, v in data.items():
            chunks.append(f"{k}: {v}")
    else:
        chunks.append(str(data))

    return chunks


def read_pdf_file(path: Path) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
    except ImportError:
        try:
            import PyPDF2 as pypdf
            reader = pypdf.PdfReader(str(path))
        except ImportError:
            raise ImportError("PDF processing requires 'pypdf'. Run: pip install pypdf")

    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append(f"--- Page {idx} ---\n{text.strip()}")
    return "\n\n".join(pages)


def ingest_file(file_path: Path) -> list[dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    filename = path.name
    results = []

    # Skip database and binary files from vectorstore document ingestion
    if ext in [".db", ".sqlite", ".sqlite3", ".bin"]:
        return []

    if ext in [".txt", ".md"]:
        text = read_text_file(path)
        if text:
            results.append({"text": text, "source": filename, "metadata": {"file_type": ext}})
    elif ext == ".csv":
        rows = read_csv_file(path)
        for row in rows:
            results.append({"text": row, "source": filename, "metadata": {"file_type": "csv"}})
    elif ext == ".json":
        items = read_json_file(path)
        for item in items:
            results.append({"text": item, "source": filename, "metadata": {"file_type": "json"}})
    elif ext == ".pdf":
        text = read_pdf_file(path)
        if text:
            results.append({"text": text, "source": filename, "metadata": {"file_type": "pdf"}})
    else:
        # Fallback to standard text reading
        text = read_text_file(path)
        if text:
            results.append({"text": text, "source": filename, "metadata": {"file_type": "text"}})

    return results
