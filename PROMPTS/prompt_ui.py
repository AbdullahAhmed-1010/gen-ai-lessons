import os
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import load_prompt
import streamlit

os.environ["HF_HOME"] = "E:/huggingface_cache"

@streamlit.cache_resource
def load_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "temperature": 0.7,
        }
    )

    return ChatHuggingFace(llm=llm)

model = load_model()

streamlit.header("Research Tool")

paper_input = streamlit.selectbox("Select Research Paper Name",
            ["Attention is All you Need", "BERT: Pre-Training of Deep Bidirectional Transformers",
             "GPT-3", "Diffusion Models"])

style_input = streamlit.selectbox("Select Explaination Style",
            ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = streamlit.selectbox("Select Explaination Length",
            ["Short", "Medium", "Long"])

template = load_prompt("template.json")



if streamlit.button("Summarize"):
    with streamlit.spinner("Generating response..."):

        chain = template | model
        result = chain.invoke({
            "paper_input": paper_input,
            "style_input": style_input,
            "length_input": length_input
        })

        streamlit.write(result.content)