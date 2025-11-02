from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash"
)

parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Annotated[Literal["positive", "negative"], Field(description="Give the sentiment of the feedback")]

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback review into positive or negative.\n{feedback}\n{format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="Write an appropiate response to this postive feedback.\n{feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Write an appropiate response to this negative feedback.\n{feedback}",
    input_variables=["feedback"]
)

classifier_chain = prompt1 | model | parser2

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positve", prompt2 | model | parser1),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser1),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "The product I bought broke within a week. It's poor quality."})

print(result)