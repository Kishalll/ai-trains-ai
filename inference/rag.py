from pathlib import Path
from typing import Any, Optional
from data_processing.embedder import Embedder
import deps


class VectorStore:
    def __init__(self, role_path: Path):
        self.role_path = Path(role_path)
        self.db_dir = self.role_path / "vectorstore"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = f"role_{self.role_path.name.replace('-', '_')}"
        self.embedder = Embedder()
        self._client = None
        self._collection = None

    def _init_db(self):
        if self._collection is not None:
            return

        if not deps.require_group("rag"):
            raise RuntimeError("RAG dependencies (chromadb, sentence-transformers) are required.")

        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(
            path=str(self.db_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"role": self.role_path.name},
        )

    @property
    def collection(self):
        self._init_db()
        return self._collection

    def add_documents(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0

        self._init_db()
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        ids = [c["id"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]

        # ChromaDB metadata values must be str, int, float, or bool
        cleaned_metadatas = []
        for meta in metadatas:
            clean = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            cleaned_metadatas.append(clean)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=cleaned_metadatas,
        )
        return len(chunks)

    def delete_by_source(self, source_name: str) -> int:
        self._init_db()
        existing = self.collection.get(where={"source": source_name})
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def clear(self):
        self._init_db()
        if self._client is not None:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"role": self.role_path.name},
            )

    def count(self) -> int:
        self._init_db()
        return self.collection.count()

    def get_source_counts(self) -> dict[str, int]:
        self._init_db()
        data = self.collection.get(include=["metadatas"])
        metas = data.get("metadatas", [])
        counts: dict[str, int] = {}
        for m in metas:
            if m and "source" in m:
                src = str(m["source"])
                counts[src] = counts.get(src, 0) + 1
        return counts

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        self._init_db()
        if self.count() == 0:
            return []

        query_embedding = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            # Cosine distance to similarity percentage approximation
            similarity = max(0.0, 1.0 - float(dist))
            hits.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "source": meta.get("source", "unknown") if meta else "unknown",
                    "similarity": round(similarity, 4),
                    "distance": round(float(dist), 4),
                }
            )

        return hits
