from rag.chunker import TextChunker
from rag.embeddings import EmbeddingModel
from rag.retriever import FaissRetriever
from rag.llm import TinyLlamaLLM


class RAGPipeline:
    """
    End-to-end Retrieval Augmented Generation pipeline with conversation memory.
    """

    def __init__(
        self,
        llm_cfg: dict,
        rag_cfg: dict,
        documents: list[str]
    ):
        self.history = []

        self.llm = TinyLlamaLLM(llm_cfg["model_path"])
        self.chunker = TextChunker(
            rag_cfg["chunk_size"],
            rag_cfg["chunk_overlap"]
        )
        self.embedder = EmbeddingModel()

        # Chunk documents
        chunks = []
        for doc in documents:
            chunks.extend(self.chunker.split(doc))

        # Embed chunks
        embeddings = self.embedder.embed_texts(chunks)

        # Build retriever
        self.retriever = FaissRetriever(embedding_dim=embeddings.shape[1])
        self.retriever.add(embeddings, chunks)

        self.top_k = rag_cfg.get("top_k", 3)
        self.max_new_tokens = llm_cfg.get("max_new_tokens", 128)

    def ask(self, query: str) -> str:
        """
        Answer a query using RAG + conversation history.
        """

        # Retrieve relevant chunks
        query_embedding = self.embedder.embed_query(query)
        retrieved = self.retriever.search(query_embedding, self.top_k)

        context = "\n\n".join(
            f"- {r['text']}" for r in retrieved
        )

        # Conversation history (last 4 turns)
        history_prompt = "\n".join(self.history[-4:])
        
        prompt = f"""You are a factual assistant.
        Answer ONLY using the retrieved context.
        If the answer is not present, say "I don't know".

        Conversation so far:
        {history_prompt}

        Retrieved context:
        {context}

        Question:
        {query}

        Answer (concise, factual):"""

        answer = self.llm.generate(
            prompt,
            max_new_tokens=self.max_new_tokens
        )

        # Update conversation memory
        self.history.append(f"User: {query}")
        self.history.append(f"Assistant: {answer}")

        return answer

