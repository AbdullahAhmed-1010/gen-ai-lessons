from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 200
    }
)

model = ChatHuggingFace(llm=llm)

streamlit.header("Research Tool")
user_input = streamlit.text_input("Enter your prompt: ")

if streamlit.button("Summarize"):
    result = model.invoke(user_input)
    streamlit.write(result.content)