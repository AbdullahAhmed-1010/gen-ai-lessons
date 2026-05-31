from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2"
)

result = embeddings.embed_query("Lionel Messi")

print(len(result))
print(result[:5])

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# for model in genai.list_models():
#     print(model.name)
#     print(model.supported_generation_methods)
#     print("-" * 50)