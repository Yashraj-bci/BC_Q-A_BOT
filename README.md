# BC Q&A BOT

Welcome to BC Q&A BOT, a Streamlit-powered chatbot that can answer questions based on your uploaded documents. This bot uses OpenAI's language models and conversation chains to provide intelligent responses.

## Installation

Make sure you have the required dependencies installed. You can install them using the following:

```bash
pip install streamlit python-dotenv PyPDF2 openai langchain
```

## Usage

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your environment variables by creating a `.env` file with your OpenAI API key:

```env
OPENAI_API_KEY=your-api-key
```

3. Run the Streamlit app:

```bash
streamlit run working4.py
```

## Features

- **Document Processing**: Upload PDF, DOCX, or TXT files and click on 'Process' to extract text.

- **Question Asking**: Ask questions related to the uploaded documents.

- **Domain Classification**: The bot determines the domain (Finance, Healthcare, or General) based on the user's questions.

- **Conversational Retrieval**: Utilizes conversation chains to provide context-aware responses.

## How It Works

1. **Document Processing**: Uploaded documents are processed to extract text using the appropriate parser for each file format.

2. **Text Chunking**: The extracted text is divided into chunks to facilitate efficient processing.

3. **Vector Store**: Embeddings of text chunks are stored in a vector database using FAISS.

4. **Domain Classification**: The bot classifies the user's question into one of three domains: Finance, Healthcare, or General.

5. **Conversational Retrieval**: Conversation chains are employed to retrieve relevant information from the vector store based on the user's question and domain.

## Note

This bot uses OpenAI's language models, so ensure you have a valid OpenAI API key in your environment variables.

Feel free to customize and extend this bot to suit your specific use case!
