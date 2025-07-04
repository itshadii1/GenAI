from typing import Literal
from pydantic import BaseModel, Field
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda

# 1️⃣ Instantiate your Llama model (zero temperature for deterministic, JSON-only output)
model = ChatOllama(
    model="llama3.2:latest",
    base_url="http://localhost:11434",
    temperature=0,
)

# 2️⃣ Simple parser for the follow-up response
parser = StrOutputParser()

# 3️⃣ Define your Pydantic schema for the classifier output
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description='Either "positive" or "negative"'
    )

# 4️⃣ Create the JSON‐only output parser
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# 5️⃣ Prompt that *only* returns raw JSON matching the schema
prompt1 = PromptTemplate(
    template=(
        "You are a JSON-only sentiment classifier.\n"
        "Classify the sentiment of the following feedback into \"positive\" or \"negative\".\n"
        "You *must* output *exactly* one JSON object matching this schema and nothing else:\n"
        "{format_instructions}\n\n"
        "Feedback:\n"
        "{feedback}"
    ),
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()},
)

# 6️⃣ Build the classifier chain: prompt → model → pydantic parser
classifier_chain = prompt1 | model | parser2

# 7️⃣ Prompts for positive / negative follow-ups
prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n\n{feedback}",
    input_variables=['feedback'],
)
prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n\n{feedback}",
    input_variables=['feedback'],
)

# 8️⃣ Branch on the parsed sentiment (two condition/runnable tuples + default)
branch_chain = RunnableBranch(
    (lambda fb: fb.sentiment == 'positive', prompt2 | model | parser),
    (lambda fb: fb.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda _: "could not find sentiment")
)

# 9️⃣ Assemble the full pipeline: classify → then branch
chain = classifier_chain | branch_chain

if __name__ == "__main__":
    feedback_text = "This is a beautiful phone"
    # invokes the classifier, then routes to the positive/negative responder
    reply = chain.invoke({'feedback': feedback_text})
    print(f"Feedback: {feedback_text!r}")
    print(f"Reply: {reply}\n")
    # visualize the chain structure
    chain.get_graph().print_ascii()
