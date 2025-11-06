from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

with open(file=r"C:\Users\user\Desktop\gen-ai-lessons-main\gen-ai-lessons-main\Chatbot\chatbot_using_hf.py", mode="r") as f_read:
    text = f_read.read()

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks)