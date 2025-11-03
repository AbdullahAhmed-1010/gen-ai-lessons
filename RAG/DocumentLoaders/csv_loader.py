from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash"
)

loader = CSVLoader(
    file_path=r"C:\Users\user\Desktop\gen-ai-lessons-main\gen-ai-lessons-main\RAG\DocumentLoaders\document.csv"
)

docs = loader.load()
print(docs)
print(docs[0].page_content)
print(docs[0].metadata)