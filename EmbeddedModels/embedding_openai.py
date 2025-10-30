from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(
    model="text-emebedding-3-large",
    dimensions=30 
)
# By default the length of embedding vector is 1536 for small models and it is 3072 for large models
# You can play around the dimension parameter as much you like (more vector means more contextual meaning)

result = embedding.embed_query("Delhi is the capital of India")

print(str(result))