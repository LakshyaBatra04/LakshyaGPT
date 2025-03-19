from ollama import ChatResponse
from ollama import chat
import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings

model_name = 
async def get_response(query):
    try :

        response: ChatResponse = await chat(model='mistral', messages=[
        {
            'role': 'user',
            'content': query ,
        },
        ])
        return response['messages']['content']

    except Exception as e:
        print(f"Error in infering from Ollama mistral for question :  + {query}")



async def extarct_pdf_text(path):
    doc = pymupdf.open(path)
    total_chunk=[]

    for page in doc :
        try:
                
            print(page)
            text = page.get_text()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = await text_splitter.split_text(text)
            total_chunk = total_chunk+chunks
        except Exception as e:
            print("Error in processing chunk + " + e)
    return total_chunk



def main():
    path = 'D:\Desktop\CoreShield\Yoga Philosophy .pdf'
    doc_pages = extarct_pdf_text(path)
    print(doc_pages)
if __name__ == '__main__':
    main()
