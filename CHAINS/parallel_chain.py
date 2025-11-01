from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model1 = ChatHuggingFace(llm=llm)

model2 = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash"
)

prompt1 = PromptTemplate(
    template="Generate short and simplified notes on {content}",
    input_variables=["content"]
)

prompt2 = PromptTemplate(
    template="Generate quiz questions on {content}",
    input_variables=["content"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document.\nNotes: {notes}\nQuiz: {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain

with open("CHAINS/content.txt", mode="r") as f_read:
    content = f_read.read()
    result = chain.invoke({"content": content})

    with open("CHAINS/notes_and_quiz.md", mode="w") as f_write:
        f_write.write(result)

chain.get_graph().print_ascii()