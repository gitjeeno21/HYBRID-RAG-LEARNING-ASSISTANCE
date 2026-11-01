**Project Goal:** Build an Autonomous RAG (Retrieval-Augmented Generation) pipeline.
**Tech Stack:** Python, LangChain, ChromaDB, Sentence-Transformers, PyPDF.
**Strict Rules:**
- We are running locally on a 4GB VRAM GPU (RTX 3050). Use lightweight embedding models like 'all-MiniLM-L6-v2'.
- Code must be modular. Break tasks into ingestion (chunking/embedding) and generation (querying).
