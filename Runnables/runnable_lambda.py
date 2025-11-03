from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash"
)

def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template="Generate a joke about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

sequential_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_count)
})

chain = RunnableSequence(sequential_chain, parallel_chain)

response = chain.invoke({"topic": "AI"})

result = '''{}\n word count: {}'''.format(response["joke"], response["word_count"])
print(result)