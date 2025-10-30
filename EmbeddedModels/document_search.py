from langchain_huggingface import HuggingFaceEmbeddings
# from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy

# load_dotenv()

embedding = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "India is known for its rich cultural heritage and diverse traditions.",  
    "The capital of India is New Delhi, which is an important political and historical center.",  
    "The Himalayas in the north are home to some of the world's highest peaks.",  
    "Technology and innovation are rapidly growing sectors in modern India.",  
    "The Indian economy is one of the fastest-growing in the world, driven by IT and manufacturing."
]

user_query = input("Enter your query: ")

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(user_query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(user_query)
print(documents[index])
print("Similarity score is:", score)