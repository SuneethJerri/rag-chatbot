from rag.pipeline import RAGPipeline
import yaml

llm_cfg = yaml.safe_load(open("configs/model.yaml"))["llm"]
rag_cfg = yaml.safe_load(open("configs/rag.yaml"))["chunking"]

documents = [
    open("data/doc1.txt").read()
]

rag = RAGPipeline(llm_cfg, rag_cfg, documents)

print("RAG Chatbot (type 'exit' to quit)\n")

while True:
    query = input("You: ")
    if query.lower() in {"exit", "quit"}:
        break

    answer = rag.ask(query)
    print("\nBot:", answer, "\n")

