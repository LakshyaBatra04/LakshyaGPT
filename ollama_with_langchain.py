import langchain_ollama
from langchain_ollama import ChatOllama

model = ChatOllama(
    model = "mistral",
    temperature = 0.7,

)
messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]
print(model.invoke(messages).content)

