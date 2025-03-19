def main():
    from agno.agent import Agent  # Move inside function
    from agno.models.ollama import Ollama
    from agno.embedder.ollama import OllamaEmbedder
    from agno.knowledge.pdf import PDFKnowledgeBase
    from agno.vectordb.pgvector import PgVector

    db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
    embeddings = OllamaEmbedder(id="deepseek-r1:1.5b")

    knowledge_base = PDFKnowledgeBase(
        path=r"D:\3rd Year 2nd Semester\HUL352\test",
        vector_db=PgVector(
            table_name="pdf_documents",
            db_url=db_url,
            embedder=embeddings
        ),
    )
    knowledge_base.load(recreate=False)

    agent = Agent(
        model=Ollama(id="mistral"),
        description="You are an AI agent at CoreShield Technologies, you answer queries",
        show_tool_calls=True,
        markdown=True
    )

    agent.print_response("What is sankhya philosophy?", stream=True)

if __name__ == "__main__":
    main()
