import os
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
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

# template
template = PromptTemplate(
    template='''
    Please summarize the research paper titled "{paper_input}" with the following specifications:
    Explanation Style: {style_input}
    Explanation Length: {length_input}
    1. Mathematical Equations:
    Include relevant mathematical equations if present in the paper.
    Explain the mathematical concepts using simple, intuitive code snippets where applicable.
    2. Use practical analogies to simplify complex ideas.
    Where sufficient information is not available in the paper, respond with “insufficient information”.
    Make sure the explanation is clear, accurate and aligned with the prescribed style and length.
    ''',
    input_variables=["paper_input", "style_input", "length_input"]
)

prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if streamlit.button("Summarize"):
    with streamlit.spinner("Generating response..."):
        result = model.invoke(prompt)
        streamlit.write(result.content)