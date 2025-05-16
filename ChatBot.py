from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


from langchain_groq import ChatGroq
model = ChatGroq(model = "Gemma2-9b-It", groq_api_key = groq_api_key)

prompt = ChatPromptTemplate.from_messages(

    [
        ('system', 'You are a trained psychologist with expertise in cases where people have lost someone very close to them and are grieving. You must act empathetically, encourage the user to open up, and provide comfort to them.'),
        MessagesPlaceholder(variable_name='messages')

    ]

)

parser = StrOutputParser()

chain = prompt | model | parser


def run_bot():
    while True:
        message = input("Please tell me how can I help you. Press 1 to exit")
        if message != "1":
            response = chain.invoke({'messages': [HumanMessage(content= f"{message}")]})
            print(response) 
        else:
            break
    
if __name__ == "__main__":
    run_bot()