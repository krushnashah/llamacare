import logging
import os
import streamlit as st
import chromadb
import numpy as np
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# File paths
VECTORSTORE_PATH = "vectorstore/chromadb"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Upgraded to 768D embeddings
LLM_MODEL = "llama3.2"

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=VECTORSTORE_PATH)

# Function to get custom prompt
def get_custom_prompt():
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are LLamacare, a friendly and professional AI medical assistant specialized in answering questions about Polycystic Ovary Syndrome (PCOS).\n"
            "Your responses must be based only on scientific research from PubMed Central articles stored in the database.\n"
            "1. Use only the retrieved information to respond accurately.\n"
            "2. If no relevant data is found, reply with: 'I do not have enough information to answer that'.\n"
            "3. Ensure responses are professional yet friendly, clear, concise, and well-structured.\n"
            "4. If necessary, ask follow-up questions to refine the user's inquiry."
        ),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Provide an accurate and well-structured response based on the given context."
        )
    ])

# Function to load ChromaDB vectorstore
def load_chromadb_database():
    logging.info("🔄 Loading ChromaDB vectorstore...")

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Try to get collection, recreate if dimension mismatch
    try:
        collection = chroma_client.get_collection(name="pcos_articles")
        stored_dim = len(collection.peek(limit=1)["embeddings"][0])
        new_dim = len(embeddings.embed_query("test"))

        if stored_dim != new_dim:
            logging.warning(f"⚠️ Dimension mismatch: Stored {stored_dim}, New {new_dim}. Recreating collection...")
            chroma_client.delete_collection("pcos_articles")
            collection = chroma_client.get_or_create_collection(name="pcos_articles")
            logging.info("✅ Created a fresh ChromaDB collection.")

    except chromadb.errors.InvalidCollectionException:
        logging.info("🆕 Creating a new ChromaDB collection...")
        collection = chroma_client.get_or_create_collection(name="pcos_articles")

    # Load Chroma as a retriever
    vector_db = Chroma(
        client=chroma_client,
        collection_name="pcos_articles",
        embedding_function=embeddings,
    )

    logging.info(f"✅ ChromaDB vectorstore loaded successfully with {collection.count()} documents.")
    return vector_db

# Function to initialize the QA Chain
def initialize_qa_chain(vector_store):
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    return RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": get_custom_prompt()}
    )

# Streamlit UI
def main():
    st.set_page_config(page_title="LLamacare - Your PCOS AI Assistant", layout="wide")

    # Custom Styling
    st.markdown("""
        <style>
            body {
                background-color: #F5F5F5;
            }
            .chat-container {
                max-width: 800px;
                margin: auto;
                padding: 20px;
                border-radius: 10px;
                background-color: #ffffff;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            }
            .user-message {
                text-align: right;
                background-color: #C3E6FC;
                padding: 12px;
                border-radius: 10px;
                margin-bottom: 10px;
                font-weight: bold;
                color: #1B4F72;
            }
            .assistant-message {
                text-align: left;
                background-color: #D4EDDA;
                padding: 12px;
                border-radius: 10px;
                margin-bottom: 10px;
                font-weight: bold;
                color: #155724;
            }
            .title {
                text-align: center;
                font-size: 40px;
                font-weight: bold;
                color: #2C3E50;
            }
            .ask-button {
                background-color: #008080 !important;
                color: white !important;
                border-radius: 8px;
                padding: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title'>LLamacare - Your PCOS AI Assistant</div>", unsafe_allow_html=True)
    st.markdown("""
        **Welcome to LLamacare, your trusted AI medical assistant for PCOS-related questions!**
        
        🩺 **Reliable Medical Insights**     |     📖 **Evidence-Based Responses**     |     💬 **Friendly and Professional Advice**
        
        Ask any PCOS-related medical question, and LLamacare will provide a science-backed answer!
    """)

    # Load ChromaDB vectorstore
    vector_store = load_chromadb_database()
    qa_chain = initialize_qa_chain(vector_store)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()
    for message in st.session_state.messages:
        role, content = message
        chat_container.markdown(f"<div class='{role}-message'>{content}</div>", unsafe_allow_html=True)

    user_question = st.text_input("Type your question here:")
    if st.button("Ask LLamacare", key="ask_button") and user_question:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"query": user_question})
            answer = response["result"]

            st.session_state.messages.append(("user", user_question))
            st.session_state.messages.append(("assistant", answer))

            chat_container.markdown(f"<div class='user-message'>{user_question}</div>", unsafe_allow_html=True)
            chat_container.markdown(f"<div class='assistant-message'>{answer}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
