import random
from abc import ABC,abstractmethod

class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass

class MyLLM(Runnable):
    def __init__(self):
        # print("LLM Created)
        pass

    def invoke(self, invoke):
        response_list = [
            "Delhi is the capital of India.",
            "AI stands for Artificial Intelligence.",
            "Earth is the 3rd planet in our solar system."
        ]       
        return {"response": random.choice(response_list)}
    
    def predict(self, prompt):
        response_list = [
            "Delhi is the capital of India.",
            "AI stands for Artificial Intelligence.",
            "Earth is the 3rd planet in our solar system."
        ]       
        return "This module is depracated now. It is advised to use invoke method",
        {"response": random.choice(response_list)}
    
class MyPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    def format(self, input_dict):
        return "This module is depracated now. It is advised to use invoke method",
        self.template.format(**input_dict)
    
class MyStrOutputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data):
        return input_data["response"]
    
class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        return input_data

template1 = MyPromptTemplate(
    template="Generate a {length} poem about {topic}",
    input_variables=["length", "topic"]
)

llm = MyLLM()

parser = MyStrOutputParser()

chain1 = RunnableConnector([template1, llm, parser])

result1 = chain1.invoke({"length": "short", "topic": "freedom"})
print(result1)

template2 = MyPromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

template3 = MyPromptTemplate(
    template="Explain the following joke.\n{response}",
    input_variables=["response"]
)

chain2 = RunnableConnector([template2, llm])

chain3 = RunnableConnector([template3, llm, parser])

final_chain = RunnableConnector([chain2, chain3])

result2 = final_chain.invoke({"topic": "AI"})
print(result2)