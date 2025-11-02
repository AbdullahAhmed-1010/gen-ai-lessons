import random

# How LLM component works?
class MyLLM:
    def __init__(self):
        print("LLM Created")

    def predict(self, prompt):
        response_list = [
            "Delhi is the capital of India.",
            "AI stands for Artificial Intelligence.",
            "Earth is the 3rd planet in our solar system."
        ]       
        return {"response": random.choice(response_list)}
    
llm = MyLLM()

result = llm.predict("What is the capital of India?")
print(result)

# How Prompt Template component works?
class MyPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)
    
template = MyPromptTemplate(
    template="Generate a {length} poem about {topic}",
    input_variables=["length", "topic"]
)

prompt = template.format({"length": "short", "topic": "freedom"})

llm = MyLLM()
llm.predict(prompt)

# How Chain component works?
class MyLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)

        return result["response"]
    
llm = MyLLM()

template = MyPromptTemplate(
    template="Generate a {length} poem about {topic}",
    input_variables=["length", "topic"]
)

chain = MyLLMChain(llm=llm, prompt=template)
chain.run({"length": "short", "topic": "freedom"})