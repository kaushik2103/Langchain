# Q&A Chatbot
from langchain.llms import OpenAI
from dotenv import load_dotenv
import streamlit as st
from constant import OPENAI_KEY
import os

os.environ["OPENAI_API_KEY"] = OPENAI_KEY
load_dotenv()


# Function to load OpenAI model and get response
def get_openai_response(question):
    llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0.5)
    return llm(question)


# Initialize the streamlit app
st.set_page_config(page_title="Q&A Chatbot", page_icon=":robot:")
st.header("Langchain Application")
question = st.text_input("Enter your question: ", key="input")
response = get_openai_response(question)

submit = st.button("Ask a question")

if submit | (bool(question)):
    st.header("The Response is: ")
    st.write(response)
