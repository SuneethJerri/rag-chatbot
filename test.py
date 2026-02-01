import yaml
from rag.pipeline import RAGPipeline

llm_cfg = yaml.safe_load(open("configs/model.yaml"))["llm"]
rag_cfg = yaml.safe_load(open("configs/rag.yaml"))["chunking"]

documents = [
    """Retrieval Augmented Generation (RAG) combines information retrieval
    with text generation. Instead of relying only on parametric knowledge,
    it retrieves relevant documents and uses them as context."""
]

rag = RAGPipeline(llm_cfg, rag_cfg, documents)

answer = rag.ask("How does RAG reduce hallucinations?")
print(answer)

