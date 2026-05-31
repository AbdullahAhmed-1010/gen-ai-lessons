from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2"
)

# create langchain documents
doc1 = Document(
    page_content="Lionel Messi",
    metadata={"Team": "Argentina"}
)
doc2 = Document(
    page_content="Neymar Jr",
    metadata={"Team": "Brazil"}
)
doc3 = Document(
    page_content="Cristiano Ronaldo",
    metadata={"Team": "Portugal"}
)
doc4 = Document(
    page_content="Lamine Yamal",
    metadata={"Team": "Spain"}
)
doc5 = Document(
    page_content="Harry Kane",
    metadata={"Team": "England"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory='chroma_db',
    collection_name='sample'
)

# add documents
vector_store.add_documents(docs)

# view documents
vector_store.get(include=["embeddings", "documents", "metadatas"])

# search documents
vector_store.similarity_search(
    query="Who among them is from argentina?",
    k=3
)
vector_store.similarity_search_with_score(
    query="Who among them is from argentina?",
    k=3
)

# metadata filtering
vector_store.similarity_search_with_score(
    query="",
    filter={"Team": "Spain"}
)

# update documents
updated_doc1 = Document(
    page_content="Paulo Dybala",
    metadata={"Team": "Argentina"}
)
vector_store.update_document(document_id="9905ada9-4e62-4230-a60c-8b4b496752be", document=updated_doc1)
vector_store.get(include=["embeddings", "documents", "metadatas"])

# delete documents
vector_store.delete(ids="9905ada9-4e62-4230-a60c-8b4b496752be")
