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

loader = WebBaseLoader(
    web_paths=["https://en.wikipedia.org/wiki/Vedanta",
               "https://en.wikipedia.org/wiki/M%C4%ABm%C4%81%E1%B9%83s%C4%81"],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(["p", "h1", "h2", "h3"])  
    ),
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

vector_storage= vector_store.add_documents(documents=all_splits)
retriever = vector_store.as_retriever()


