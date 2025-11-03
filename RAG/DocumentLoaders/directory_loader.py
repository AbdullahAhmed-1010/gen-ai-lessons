from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash"
)

loader = DirectoryLoader(
    path=r"C:\Users\user\Desktop\gen-ai-lessons-main\gen-ai-lessons-main\RAG\DocumentLoaders\documents",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()
print(type(docs))
print(docs[0].page_content)
print(docs[0].metadata)