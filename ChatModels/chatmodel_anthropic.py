from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(
    model="claude-3-5-sonnet-2020012",
    temperature=0,
    max_tokens_to_sample=10
)

result = model.invoke("What is the capital of India?")

print(result.content)