from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash"
)

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following report.\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

sequential_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(sequential_chain, branch_chain)

result = chain.invoke({"topic": "USSR"})
print(result)