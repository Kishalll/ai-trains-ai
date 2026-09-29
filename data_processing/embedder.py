from typing import Optional
import deps

_MODEL_INSTANCE = None
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    if not deps.require_group("rag"):
        raise RuntimeError("RAG dependencies (sentence-transformers, chromadb) are required to generate embeddings.")

    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model '{model_name}'...")
    _MODEL_INSTANCE = SentenceTransformer(model_name)
    return _MODEL_INSTANCE


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        # Pre-load the embedding weights upfront so queries never encounter cold starts
        get_embedding_model(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = get_embedding_model(self.model_name)
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> list[float]:
        model = get_embedding_model(self.model_name)
        embedding = model.encode(query, convert_to_numpy=True, show_progress_bar=False)
        return embedding.tolist()
