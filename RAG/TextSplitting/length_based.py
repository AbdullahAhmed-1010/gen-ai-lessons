from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    r"C:\Users\user\Desktop\gen-ai-lessons-main\gen-ai-lessons-main\RAG\DocumentLoaders\document.pdf"
)

# text = '''
# A black hole is one of the most mysterious and fascinating objects in the universe, representing the ultimate
# triumph of gravity over matter and energy. It forms when a massive star exhausts its nuclear fuel and collapses
# under its own gravity, compressing all its mass into an incredibly small, infinitely dense point known as a
# singularity. Surrounding this singularity is the event horizon, the boundary beyond which nothing—not even
# light—can escape the immense gravitational pull. Because no light can escape, black holes appear completely
# dark, which is why they are called “black.” Despite their invisibility, scientists can detect black holes
# by observing how they affect nearby matter and light. For instance, when a black hole pulls in
# surrounding gas and dust, the material spirals inward at tremendous speeds, heating up and emitting
# powerful X-rays before crossing the event horizon.
# '''

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ""
)

# result = splitter.split_text(text)
result = splitter.split_documents(docs)
print(result[0].page_content)