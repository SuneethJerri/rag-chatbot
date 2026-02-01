from sentence_transformers import SentenceTransformer
import torch


class EmbeddingModel:
    """
    Wrapper around a sentence-transformers embedding model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )

    def embed_texts(self, texts: list[str]):
        """
        Embed a list of texts into vectors.
        Returns a list of numpy arrays.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings

    def embed_query(self, query: str):
        """
        Embed a single query string.
        """
        return self.embed_texts([query])[0]

