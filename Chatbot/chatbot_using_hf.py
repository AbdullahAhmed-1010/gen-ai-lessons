import os
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

os.environ["HF_HOME"] = "E:/huggingface_cache"

def load_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "temperature": 0.5
        }
    )
    return ChatHuggingFace(llm=llm)

model = load_model()

def generate_prompt(chat_history, user_input):
    prompt = ""
    for user_msg, assistant_msg in chat_history:
        prompt = prompt + f"<|user|>\n{user_msg}\n</s><|assistant|>\n{assistant_msg}\n</s>"
    prompt = prompt + f"<|user|>\n{user_input}\n</s><|assistant|>"
    return prompt

chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    # prompt = generate_prompt(chat_history, user_input)
    # result = model.invoke(prompt)
    # print("AI:", result.content)
    # # Extract the true reply after <|assistant|> to </s>
    # reply = str(result.content).split('<|assistant|>')[-1].split('</s>')[0].strip()
    # chat_history.append((user_input, reply))

print(chat_history)