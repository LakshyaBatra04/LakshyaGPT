from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.embedder.ollama import OllamaEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools

from agno.models.ollama import Ollama
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

embeddings = OllamaEmbedder(id="mistral")

# Set up the knowledge base with the Hugging Face embedder
knowledge_base = PDFKnowledgeBase(
    path=r"D:\3rd Year 2nd Semester\HUL352\test",
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url=db_url,
        embedder=embeddings  # Use the Hugging Face embedder
    ),
)
knowledge_base.load(recreate=False)

# Initialize the agent with the desired model
agent = Agent(
    model=Ollama(id="mistral"),
    knowledge=knowledge_base,
    system_message="You are an AI agent at CoreShield Technologies, answer to the point and quickly to the user's query ",
    search_knowledge=True,
    tools=[DuckDuckGoTools()],
    # Add a tool to read chat history.
    show_tool_calls=True,
    debug_mode=True,
    markdown=True
)

# Get the agent's response to the query
agent.print_response(
    "Tell me about the recent ICC Champions Trophy?", stream=True
)
