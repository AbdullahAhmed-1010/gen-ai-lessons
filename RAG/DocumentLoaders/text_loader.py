from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash"
)

prompt = PromptTemplate(
    template="Write a summary of the following document.\n{document}",
    input_variables=["document"]
)

parser = StrOutputParser()

loader = TextLoader(
    r"C:\Users\user\Desktop\gen-ai-lessons-main\gen-ai-lessons-main\RAG\DocumentLoaders\document.txt",
    encoding="utf-8"
)

docs = loader.load()
print(docs)
print(type(docs))
print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt | model | parser

result = chain.invoke({"document": docs[0].page_content})
print(result)