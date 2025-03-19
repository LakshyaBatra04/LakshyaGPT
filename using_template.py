from langchain_core.prompts import PromptTemplate
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
query = "What is vedanta episotmology?"
docs = vector_store.similarity_search(query, k=1)
context= ""
for doc in docs:
    context+='\n'
    context = context+' ' + doc.page_content
    
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    '''User Query : {query}
    context : {context}
    #You are given a user query and the context required to answer the query.
    # Your job is to make a prompt for another LLM model, asking it to answer the query 
    # using the context provided and its own knowlegde. You should not answer the query yourself. Just provide the 
    # prompt for the other LLM
    # NO PREAMBLE, JUST THE PROMPT'''
)

refined_prompt = chat(model='mistral', messages=[{'role': 'user', 'content': template.format(query=query, context=context)}])

# Extract the generated refined prompt

mistral_generated_prompt = refined_prompt['message']['content']
print('mistral prompt' + mistral_generated_prompt)

print("--------------------------")
final_response = chat(model='mistral_tuned',
                       messages=[{'role': 'user', 'content': mistral_generated_prompt}]
                    )

print('final ans : '+ final_response['message']['content'])


