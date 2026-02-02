# RAG Chatbot

A modular, high-performance Retrieval Augmented Generation (RAG) chatbot with YAML-driven configuration and 4-bit quantization optimization. It grounds responses in indexed documents and maintains conversation memory.

## Features

- **Document-grounded answers** using FAISS similarity search
- **Conversation memory** (last 4 turns)
- **YAML configuration** for model and RAG parameters
- **4-bit quantization** to reduce memory and latency
- **Modular components**: chunker, embedder, retriever, LLM, orchestrator

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place documents under `data/` (e.g., `data/doc1.txt`).

3. Configure paths and parameters in `configs/model.yaml` and `configs/rag.yaml`.

4. Run the interactive chat:
   ```bash
   python chat.py
   ```

5. (Optional) Benchmark FP16 vs 4-bit performance:
   ```bash
   python benchmark.py
   ```

## Architecture

The system follows a two-phase design: offline indexing at startup and fast online querying per request. The `RAGPipeline` orchestrates `TextChunker`, `EmbeddingModel`, `FaissRetriever`, and `TinyLlamaLLM` components.

### Offline Indexing
- Documents are split into overlapping chunks via `TextChunker` .
- Chunks are embedded by `EmbeddingModel` using `all-MiniLM-L6-v2`.
- Embeddings are indexed in `FaissRetriever` with inner-product similarity .

### Online Querying
- Query is embedded and top-k chunks are retrieved.
- Retrieved context and conversation history are assembled into a prompt .
- `TinyLlamaLLM` generates the answer.
- Conversation memory is updated with the turn.

## Configuration

- `configs/model.yaml`: `model_path`, `max_new_tokens`, and quantization settings.
- `configs/rag.yaml`: `chunk_size`, `chunk_overlap`, `top_k`.

Both are loaded in `chat.py` and passed to `RAGPipeline`.

## Optimization: 4-bit Quantization

`benchmark.py` demonstrates 4-bit quantization using `BitsAndBytesConfig` with `load_in_4bit=True` and `bnb_4bit_quant_type="nf4"` [12](#1-11) . The config is passed to `AutoModelForCausalLM.from_pretrained`.

### Benchmark Report

Running `python benchmark.py` measures and prints:
- Load time (ms)
- Generation time (ms)
- Peak GPU memory (MB)

It outputs a side-by-side summary for FP16 and 4-bit modes.

## Usage Examples

### Interactive Chat
```bash
python chat.py
You: What is retrieval augmented generation?
Bot: Retrieval Augmented Generation (RAG) is a technique that combines information retrieval with language model generation...
```

### Benchmark
```bash
python benchmark.py
Running FP16 baseline...
...
Running 4-bit optimized...
...
=== Summary ===
FP16     : (load_ms, gen_ms, mem_mb)
4-bit    : (load_ms, gen_ms, mem_mb)
```

## Development

- Add new documents to `data/`.
- Tune chunking/retrieval via `configs/rag.yaml`.
- To enable quantization in chat, extend `rag/llm.py` to accept a quantization flag similar to `benchmark.py`.

## Notes

- The chat interface currently runs FP16; quantization is demonstrated in `benchmark.py`.
- The benchmark report is console output; redirect to a file or embed in README as needed.
- Components are modular and single-responsibility for easy upgrades.

Wiki pages you might want to explore:
- [Overview (SuneethJerri/rag-chatbot)](/wiki/SuneethJerri/rag-chatbot#1)
- [System Architecture (SuneethJerri/rag-chatbot)](/wiki/SuneethJerri/rag-chatbot#1.1)


