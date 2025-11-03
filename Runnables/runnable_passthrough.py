from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash"
)

prompt1 = PromptTemplate(
    template="Generate a joke about {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke.\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

generator_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(prompt2, model, parser)
})

chain = RunnableSequence(generator_chain, parallel_chain)

result = chain.invoke({"topic": "AI"})
print(result)