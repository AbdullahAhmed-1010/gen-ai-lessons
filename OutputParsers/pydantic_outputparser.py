from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash"
)

class Person(BaseModel):
    name: Annotated[str, Field(description="Name of the person")]
    age: Annotated[int, Field(gt=18, description="Age of the person")]
    city: Annotated[str, Field(description="Name of the city that person belongs to")]

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {country} person \n{format_instruction}",
    input_variables=["country"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"country": "japan"})

print(result)