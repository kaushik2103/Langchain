# Integrate our code OpenAI API

import os
from constant import openai_key
from langchain.llms import OpenAI
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# Initialize Streamlit Framework
st.title("Langchain Demo with OpenAI API")
input_text = st.text_input("Search the topic you want to search: ")

# OpanAI LLM Model
llm = OpenAI(temperature=0.8)

if input_text:
    st.write(llm(input_text))
