import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient

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


import requests

API_URL = "https://router.huggingface.co/hf-inference/v1"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Can you please let us know more details about your ",
})

print(output)
