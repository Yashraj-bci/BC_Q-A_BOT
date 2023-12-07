import streamlit as st
import openai
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
#from docx import Document  # Add missing import for Document
import os
from datetime import datetime
import logging

_ = load_dotenv(find_dotenv()) # read local .env file


logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

st.set_page_config(page_title="Multi-Domain Chatbot", page_icon=":robot_face:")

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if 'current_domain' not in st.session_state:
    st.session_state.current_domain = "General"

question = st.text_input("User Input", key="user_input", value="", help="Enter your message...")

# Define domain-specific prompts for finance and healthcare
finance_prompt = """I'm a finance expert. Ask me anything about investing, income taxes, accounting, or other financial topics."""
healthcare_prompt = """I'm a healthcare expert. Ask me about medical conditions, treatments, or other healthcare topics."""

# Define domain-specific question answering models
#finance_model = ChatOpenAI(prompt=finance_prompt)
#healthcare_model = ChatOpenAI(prompt=healthcare_prompt)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#def get_docx_text(docx_docs):
    text = ""
    for docx in docx_docs:
        doc = Document(docx)
        for para in doc.paragraphs:
            text += para.text + '\n'
    return text

def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        text += txt.getvalue().decode() + '\n'
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, domain):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    if domain == "Finance":
        llm = finance_model
    else:
        llm = healthcare_model
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.current_domain == "Finance":
        response = st.session_state.conversation({'question': user_question})
    elif st.session_state.current_domain == "Healthcare":
        response = st.session_state.conversation({'question': user_question})
    else:
        healthcare_response = healthcare_model.generate(user_question)
        finance_response = finance_model.generate(user_question)
        if len(healthcare_response) > len(finance_response):
            response = healthcare_model.generate(user_question)
            st.session_state.current_domain = "Healthcare"
        else:
            response = finance_model.generate(user_question)
            st.session_state
