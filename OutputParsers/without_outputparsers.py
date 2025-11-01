import os
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

os.environ["HF_HOME"] = "E:/huggingface_cache"

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.7
    }
)

model = ChatHuggingFace(llm=llm)

# Detailed report
template = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# Summary
new_template = PromptTemplate(
    template="Write a 5 line summary on the following text.\n{text}",
    input_variables=["text"]
)

user_query = {"topic": "black hole"}
prompt = template.invoke(user_query)

result = model.invoke(prompt)

new_prompt = new_template.invoke({"text": result.content})

response = model.invoke(new_prompt)

print(response)