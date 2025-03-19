from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.embedder.ollama import OllamaEmbedder
from agno.models.ollama import Ollama
from agno.tools.duckduckgo import DuckDuckGoTools

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

embeddings = OllamaEmbedder(id="mistral")

knowledge_base = PDFKnowledgeBase(
    path=r"D:\3rd Year 2nd Semester\HUL352\test",
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url=db_url,
        embedder=embeddings 
    ),
    num_documents= 2    
)
knowledge_base.load(recreate=False)

agent = Agent(
    model=Ollama(id="mistral", show_tool_calls = True),
    knowledge=knowledge_base,
    system_message="You are an AI agent at CoreShield Technologies, you answer queries from the documents provided. "
    "If the query's answer is not in the documents provided, simply say 'the answer is not in the provided documents', do not make up your own answer ",
    add_references=True,
    search_knowledge=True,
    read_chat_history=True,
    show_tool_calls=True,
    markdown=True
)

while True:
    query = input()
    if query.lower() in ["exit"]:
        break  
    agent.print_response(
        query, stream=True
    )
print(agent.memory)