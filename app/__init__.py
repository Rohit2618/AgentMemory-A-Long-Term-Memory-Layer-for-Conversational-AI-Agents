"""AgentMemory — a long-term memory layer for conversational AI agents.

Package layout (mirrors the pipeline stages):

    app/
      core/           config, domain models, logging
      llm/            LLM provider abstraction (Groq + mock)
      embeddings/     embedding provider abstraction (local + mock)
      ingestion/      conversation ingestion + relational storage
      extraction/     LLM-based fact extraction
      consolidation/  conflict/update/merge decision engine (the novel piece)
      storage/        graph store (Neo4j / NetworkX) + vector store (Chroma)
      retrieval/      hybrid retrieval (vector + graph traversal)
      agent/          end-to-end chat agent using the memory system
      eval/           evaluation harness + synthetic dataset
      api/            FastAPI app wiring everything together
"""

__version__ = "0.1.0"
