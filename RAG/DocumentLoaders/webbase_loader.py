from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash"
)

prompt = PromptTemplate(
    template="Answer the following question\n{question} from the text: \n{text}",
    input_variables=["question", "text"]
)

parser = StrOutputParser()

url = "https://en.wikipedia.org/wiki/Black_hole"
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({"question":"what is a black hole?", "text": docs[0].page_content})
print(result)