from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

chat_template = ChatPromptTemplate([
        ("system", "You are a helpful assistent"),
        (MessagesPlaceholder(variable_name="chat_history")),
        ("human", "{query}"),
    ])

chat_history = [
    HumanMessage(
        content="Tell me about the Earth.", additional_kwargs={}, response_metadata={}
    ),
    AIMessage(
        content="Earth is the third planet from the Sun and the only known planet to support life, thanks to its ideal temperature, liquid water, and life-sustaining atmosphere. It is a rocky planet with a surface that is about 70% water and 30% land, and its interior is made up of a crust, mantle, and core. The Earth's rotation causes day and night, and its ~365-day orbit around the sun gives us a year.",
        additional_kwargs={},
        response_metadata={},
    ),
]


user_input = "Shorten the explanation."
prompt = chat_template.invoke({"chat_history": chat_history, "query": user_input})

print(prompt)