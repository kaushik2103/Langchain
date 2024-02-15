from PyPDF2 import PdfReader
from transformers import BertModel, BertTokenizer
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from typing_extensions import Concatenate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
import streamlit as st  

import os
from constant import HUGGINGFACE_KEY

os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_KEY

# Read PDF file
pdfreader = PdfReader("Medical_Book.pdf")
raw_text = ""
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# We need to use the text splitter to split the text into smaller chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=3000,
    chunk_overlap=5,
    length_function=len,
)
text = text_splitter.split_text(raw_text)

# Embedding the text with Hugging Face (using BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Obtain embeddings
embeddings = []
for txt in text:
    inputs = tokenizer(txt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    pooled_output = outputs.pooler_output
    embeddings.append(pooled_output)

# Convert embeddings to FAISS index
documents_search = FAISS.from_vectors(embeddings)

# Chains to answer questions
chain = load_qa_chain(HuggingFaceHub(), chain_type="stuff")

# Streamlit app to query the chain with user input
st.title("Medical Disease Diagnosis and Treatment")
query = st.text_input("Enter your question: ")

if query:
    docs = documents_search.similarity_search(query)
    st.write(chain.run(input_documents=docs, question=query))
