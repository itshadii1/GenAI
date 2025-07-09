from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama

model = ChatOllama(
    model="llama3.2:latest",
    base_url="http://localhost:11434",
    temperature=0,
)

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model | parser
#metadata and page_content can be accessed ex: docs[0].metadeta
print(chain.invoke({'poem':docs[0].page_content}))

