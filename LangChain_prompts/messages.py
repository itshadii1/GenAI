from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Intelligent-Internet/II-Medical-8B-1706",
    task="text-generation",
    
)
model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content='You are a helpful assisstant'),
    HumanMessage(content='Tell me about LangChain')
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)