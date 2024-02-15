# Conversational Chatbot with Memory to Remember

import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import os
from constant import OPENAI_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_KEY

st.set_page_config(
    page_title="Conversational Chatbot with Memory to Remember",
    page_icon=":robot:"
)
st.header("Conversational Chatbot with Memory to Remember")

chat_openai = ChatOpenAI(temperature=0.5, openai_api_key=OPENAI_KEY)

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="You are a comedian AI assistent")
    ]


def get_openai_response(user_input):
    st.session_state['flowmessages'].append(
        HumanMessage(content=user_input)
    )
    answer = chat_openai(
        st.session_state['flowmessages']
    )
    st.session_state['flowmessages'].append(
        AIMessage(content=answer.content)
    )
    return answer.content

input_text = st.text_input("Your Question: ")
response = get_openai_response(input_text)
submit = st.button("Submit")

if submit | bool(input_text):
    st.subheader("Conversation:")
    st.write(response)

