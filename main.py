import ollama
from ollama import chat
from ollama import ChatResponse
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings= HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
PERSIST_DIR = "./chroma_db"

vector_store = Chroma(persist_directory=PERSIST_DIR,embedding_function=embeddings)
query = "What is mismasa episotmology?"
docs = vector_store.similarity_search(query, k=1)
context= ""
for doc in docs:
    context+='\n'
    context = context+' ' + doc.page_content
  
response: ChatResponse = chat(model='mistral', messages=[
  {
    'role': 'user',
    'content': query ,
  },
])
print(response['message']['content'])