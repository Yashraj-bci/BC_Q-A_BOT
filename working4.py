import streamlit as st
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
#from docx import Document
import openai
import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter  # Importing CharacterTextSplitter
from htmlTemplates import css, bot_template, user_template

load_dotenv(find_dotenv())  # read local .env file
#client = OpenAI()
# Initialize OpenAI with your API key
openai.api_key = os.getenv('OPENAI_API_KEY')
# The rest of your code remains the same...

# The rest of your code remains the same...

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# def get_docx_text(docx_docs):
#     text = ""
#     for docx in docx_docs:
#         doc = Document(docx)
#         for para in doc.paragraphs:
#             text += para.text + '\n'
#     return text
#
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
    print(f'@@@@@@@@@@@@@@@@@@@{vectorstore}')
    return vectorstore

def get_doc_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_conversation_chain(vectorstore , domain):
    # Define domain-specific prompts for finance and healthcare
    finance_prompt = """I'm a finance expert. Ask me anything about investing, income taxes, accounting, or other financial topics."""
    healthcare_prompt = """I'm a healthcare expert. Ask me about medical conditions, treatments, or other healthcare topics."""
    general_prompt = """I'm a general chatbot. I'll answer every query truthfully."""

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    if domain == "Finance":
        st.sidebar.info(finance_prompt)
        print(finance_prompt)
        # openai.api_key = os.getenv('OPENAI_API_KEY')
        llm = OpenAI(model = "text-davinci-003")
        print(llm)
        
    elif domain == "Healthcare":
        st.sidebar.info(healthcare_prompt)
        print(healthcare_prompt)
        llm = ChatOpenAI()
        print(llm)
    else:
        st.sidebar.info(general_prompt)
        print(general_prompt)
        llm = ChatOpenAI()
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def determine_domain_with_llm(question):
    try:
        prompt = f"The following question is about which domain? Finance, Healthcare, or General? Strictly answer in one word.\n\nQuestion: {question}\n\nDomain:"
        openai.api_key = os.getenv('OPENAI_API_KEY')
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=1,
            n=1,
            stop=None,
            temperature=0
        )

        # Check if the response has choices
        if response.choices and response.choices[0].text:
            predicted_domain = response.choices[0].text.strip().capitalize()
            
            # Validate if predicted domain matches our domain list
            if predicted_domain in ["Healthcare", "Finance", "General"]:
                return predicted_domain

        # If no valid domain is predicted, default to "General"
        return "General"
    except openai.error.OpenAIError as e:
        st.error(f"An error occurred with the OpenAI API: {e}")
        print(f'{e}----------   ---------------------   -------------')
        return "General"

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    user_input = ''
    load_dotenv()
    st.set_page_config(page_title="BC Q&A BOT",
                        page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None  # or any other appropriate default value
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None  # or any other appropriate default value

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader(
            "Upload your documents here (PDF, DOCX, TXT) and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"]
        )

        if st.button("Process"):

            with st.spinner("Processing"):
                raw_text = ""
                
                for uploaded_file in uploaded_files:
                    if uploaded_file.type == "application/pdf":
                        raw_text += get_pdf_text([uploaded_file])
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        raw_text += get_docx_text([uploaded_file])
                    elif uploaded_file.type == "text/plain":
                        raw_text += get_txt_text([uploaded_file])

                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                domain = determine_domain_with_llm("")
                st.session_state.conversation = get_doc_chain(vectorstore)

    st.header("BC Q&A BOT :books:")
    with st.form(key="question_form"):
        user_question = st.text_input("Ask a question about your documents:")

        if st.form_submit_button("Enter"):
            
            if user_question:
                print(f'{user_question}-------------------------------------------------')
                handle_userinput(user_question)
                user_input = user_question
                print(f'{user_question}-------------------------------------------------')
                
                raw_text = ""
                
                for uploaded_file in uploaded_files:
                    if uploaded_file.type == "application/pdf":
                        raw_text += get_pdf_text([uploaded_file])
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        raw_text += get_docx_text([uploaded_file])
                    elif uploaded_file.type == "text/plain":
                        raw_text += get_txt_text([uploaded_file])

                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                domain = determine_domain_with_llm(user_input)
                st.session_state.conversation = get_conversation_chain(vectorstore, domain)

if __name__ == '__main__':
    main()
