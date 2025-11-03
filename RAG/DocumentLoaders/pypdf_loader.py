from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash"
)

prompt = PromptTemplate(
    template="Write a summary of the following text.\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

loader = PyPDFLoader(
    r"C:\Users\user\Desktop\gen-ai-lessons-main\gen-ai-lessons-main\RAG\DocumentLoaders\document.pdf"
)

docs = loader.load()
# print(docs)
# print(type(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)

chain = prompt | model | parser

result = chain.invoke({"text": docs[1].page_content})
print(docs[1].page_content)
print(result)