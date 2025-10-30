from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(
    model="text-emebedding-3-large",
    dimensions=30 
)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
] 

result = embedding.embed_documents(documents)

print(str(result))