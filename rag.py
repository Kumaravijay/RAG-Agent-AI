import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

key = os.getenv('Google_API_Key')     #Get the api key from the .env file
genai.configure(api_key=key)   #Configure the API key

gemini_model = genai.GenerativeModel('gemini-2.0-flash')  #Initialize the Gemini model

def load_embedding():
    return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')  #Load the embedding model
with st.spinner('Loading Embedding Resources...'):
    embedding_model = load_embedding()  #Load the embedding model
    
def show_gemini_styled_header():
    """
    This function injects custom HTML and CSS to display a
    Gemini-themed header in the Streamlit app.
    """
    
    html_code = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');

        /* Main container with the dark background */
        .gemini-container {
            font-family: 'Google Sans', sans-serif;
            background-color: #0E1117; /* Dark background */
            color: #E5E7EB; /* Light gray for body text */
            padding: 0.2rem;
            border-radius: 1rem;
            text-align: center;
            max-width: 600px;
            margin: 2rem auto; /* Center the container */
        }

        /* Main Headline */
        .gemini-container h1 {
            color: #FFFFFF; /* White for the main headline */
            font-size: 2.25rem; /* 36px */
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        /* Sub-headline */
        .gemini-container h2 {
            color: #9CA3AF; /* A slightly dimmer gray for contrast */
            font-size: 1.25rem; /* 20px */
            font-weight: 500;
            margin-bottom: 1.5rem;
        }

        /* Description text */
        .gemini-container p {
            font-size: 1.1rem; /* 18px */
            line-height: 1.6;
        }

        /* The special blue color for highlighting text */
        .gemini-blue {
            color: #60A5FA; /* A nice, bright Gemini blue */
            font-weight: 500;
        }
    </style>

    <div class="gemini-container">
        <h1>Gemini RAG: The <span class="gemini-blue">Ultimate</span> Document Intelligence.</h1>
        <h2>Go beyond simple keyword search.</h2>
        <p>
            Leveraging a state-of-the-art RAG architecture and <span class="gemini-blue">Gemini's reasoning</span>, this AI doesn't just search your docs—it <span class="gemini-blue">understands</span> them. Get unparalleled insights. 🚀🧠
        </p>
    </div>
    """
    
    # Use st.markdown to render the HTML
    st.markdown(html_code, unsafe_allow_html=True)

# --- Example of how to use it in your main app ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Gemini RAG")
    
    # Set a dark theme for the rest of the app to match
    st.markdown("""
        <style>
            .stApp {
                background-color: #0E1117;
            }
        </style>
    """, unsafe_allow_html=True)

    # Call the function to display the styled header
    show_gemini_styled_header()
    
    
# --- UI Rendering ---

# --- Sidebar with Creator Details (New Expander Style) ---
with st.sidebar:
    st.title("📄 App Info")
    st.info("This app uses Gemini and LangChain for Retrieval-Augmented Generation.", icon="🤖")
    # Using an expander for a cleaner look
    with st.expander("👤 About the Creator", ):
        # Replace with your details
        st.markdown("""
        **Created by:**             
        [Kumara Vijay M G]
        
        Feel free to connect for further details or collab opportunities!
        
        ---
        
        📧 **Email:** [Kumaravijay2626@gmail.com](mailto:kumaravijay2626@gmail.com)
        
        🔗 **LinkedIn:** [linkedin.com/in/Kumaravijay](https://www.linkedin.com/in/kumara-vijay-m-g-71a639317/)
        
        🐙 **GitHub:** [github.com/Kumaravijay](https://github.com/Kumaravija)
        """)
    



uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], help="Please upload a PDF document for analysis.")  #File uploader for PDF files
        
if uploaded_file:
    pdf = PdfReader(uploaded_file) 
    raw_text = ''
        
    for page in pdf.pages:
        raw_text += page.extract_text()        
    if raw_text.strip():
        doc = Document(page_content=raw_text, metadata={"source": uploaded_file.name})  #Create a Document object with the extracted text
        character_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #Initialize the text splitter    
        docs = character_text_splitter.split_documents([doc])            
            

        text = [i.page_content for i in docs]
        vector_store = FAISS.from_texts(text, embedding_model)
        retriver = vector_store.as_retriever()  
        question = st.text_input("Ask a question about the document:", key="question")
        st.button("Process Documents",key="process_button", help="Click to process the uploaded document and ask questions.")  #Button to process the documents
            
        if question:
            with st.chat_message("user"):
                with st.spinner("Generating Response... Please wait. ⏳"):
                    retrieved_docs = retriver.get_relevant_documents(question)
                    content = '\n\n'.join([i.page_content for i in retrieved_docs])
                    
                    prompt = f"Answer the question based on the context provided below:\n\n{content}\n\nQuestion: {question}"
                    content = {content}
                    question = {question}
                    response = gemini_model.generate_content(prompt)
                    st.chat_message("assistant").write(response.text)
    else:
        st.warnings("No text found in the PDF. Please upload a valid PDF file with text content.")
          