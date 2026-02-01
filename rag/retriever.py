import faiss
import numpy as np


class FaissRetriever:
    """
    FAISS-based vector retriever for RAG.
    """

    def __init__(self, embedding_dim: int):
        # Using Inner Product because embeddings are normalized
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.text_chunks = []

    def add(self, embeddings: np.ndarray, chunks: list[str]):
        """
        Add document chunk embeddings to the index.
        """
        assert len(embeddings) == len(chunks), "Embeddings and chunks size mismatch"

        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Retrieve top-k most similar chunks for a query.
        """
        query_embedding = np.expand_dims(query_embedding, axis=0)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.text_chunks[idx],
                "score": float(score)
            })

        return results

