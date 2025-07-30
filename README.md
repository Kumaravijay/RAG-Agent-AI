# Gemini RAG Agent 🤖
## An intelligent document analysis tool that allows you to have a conversation with your PDF files. This application is built using a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers directly from your uploaded documents.
### Live Demo: https://gemini-rag-kumara-vijay.streamlit.app/

## ✨ Features
#### 1. Chat with your PDFs: Upload one or more PDF documents and ask questions in natural language.

#### 2. Intelligent Answers: Powered by Google's Gemini model for high-quality, contextual responses.

#### 3. Modern UI: A clean and intuitive interface built with Streamlit.

#### 4. Efficient Processing: Uses FAISS for efficient similarity search over document embeddings.

## 🛠️ Tech Stack
#### 1. LLM: Google Gemini Pro

#### 2. Framework: Streamlit

#### 3. Language: Python

#### 4. Core Libraries: LangChain, PyPDF, FAISS, Hugging Face Transformers

## 🚀 How It Works
#### 1. Document Upload: Users upload PDF files through the Streamlit interface.

#### 2. Text Extraction & Chunking: The application extracts text and splits it into manageable chunks.

#### 3. Embedding: Each chunk is converted into a numerical vector using a sentence-transformer model.

#### 4. Indexing: The embeddings are stored in a FAISS vector store for fast retrieval.

#### 5. RAG Pipeline: When a user asks a question, the app retrieves the most relevant document chunks and feeds them, along with the question, to the Gemini model to generate a final answer.
