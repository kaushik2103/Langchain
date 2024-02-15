# Query PDF Langchain

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from typing_extensions import Concatenate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import streamlit as st

import os
from constant import OPENAI_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# Read PDF file
pdfreader = PdfReader("D:/Machine_Learning/Langchain/Medical_Chatbot/Medical_Book.pdf")
raw_text = ""
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Printing the content of the PDF file
# print(raw_text)

# We need to use the text splitter to split the text into smaller chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=10,
    length_function=len,
)
text = text_splitter.split_text(raw_text)

# Printing the text
# print(len(text))

# Embedding the text with OpenAI
embeddings = OpenAIEmbeddings()
documents_search = FAISS.from_texts(text, embeddings)

# Chains to answer questions
chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Streamlit app to query the chain with user input
st.title("Medical Chatbot App")
query = st.text_input("Enter your question: ")

if query:
    docs = documents_search.similarity_search(query)
    st.write(chain.run(input_documents=docs, question=query))


# Querying the chain
# query = "Matsya Sampada export "
# docs = documents_search.similarity_search(query)
# print(chain.run(input_documents=docs, question=query))
