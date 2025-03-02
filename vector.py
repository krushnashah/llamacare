import os
import json
import numpy as np
import faiss
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# File Paths
json_folder = "articles"
vectorstore_path = "vectorstore/db_faiss"
embedding_model = "sentence-transformers/all-MiniLM-L12-v2"

def preprocess_text(text):
    """Cleans and normalizes text for better similarity search."""
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\d+", "", text)  # Remove numbers

    text = text.replace("bp", "blood pressure")
    text = text.replace("pcos", "polycystic ovary syndrome")
    text = text.replace("t2dm", "type 2 diabetes mellitus")

    return text

def load_json_as_documents(folder_path):
    """Loads JSON files and extracts relevant content."""
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)
                tables_content = " ".join([table.get("content", "") for table in data.get("tables", [])])
                figures_content = " ".join([figure.get("caption", "") for figure in data.get("figures", [])])

                content = f"""
                    Title: {data.get("title", "No Title")}
                    Abstract: {data.get("abstract", "No Abstract")}
                    Body: {data.get("body", "No Body")}
                    Tables: {tables_content}
                    Figures: {figures_content}
                """
                content = preprocess_text(content)

                if len(content.strip()) > 10:
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "title": data.get("title", "No Title"),
                                "journal": data.get("journal", "No Journal"),
                                "publication_date": data.get("publication_date", "No Date"),
                                "source_file": file_name,
                            },
                        )
                    )

    return documents

def normalize_embeddings(embeddings):
    """Normalizes embeddings for improved similarity search."""
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def create_faiss_vectorstore(documents, embedding_model, vectorstore_path, batch_size=500):
    """Creates a FAISS vector store using IndexIVFFlat, optimized for large datasets."""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    document_texts = [doc.page_content for doc in documents]
    num_documents = len(document_texts)
    
    sample_embedding = embeddings.embed_query("sample text")
    dimension = len(sample_embedding)

    num_clusters = min(1024, num_documents // 10)  
    quantizer = faiss.IndexFlatIP(dimension)  
    index = faiss.IndexIVFFlat(quantizer, dimension, num_clusters, faiss.METRIC_INNER_PRODUCT)

    print(f"Processing {num_documents} documents in batches of {batch_size}...")

    for i in range(0, num_documents, batch_size):
        batch_texts = document_texts[i : i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch_texts)

        batch_embeddings = np.array(batch_embeddings)
        batch_embeddings = normalize_embeddings(batch_embeddings)

        if not index.is_trained:
            index.train(batch_embeddings)

        index.add(batch_embeddings)
        print(f"Processed batch {i // batch_size + 1} / {num_documents // batch_size + 1}")

        faiss.write_index(index, os.path.join(vectorstore_path, "index.faiss"))

    print(f"FAISS vectorstore saved at: {vectorstore_path}")

def test_faiss_vectorstore(vectorstore_path, embedding_model, query="Test query"):
    """Tests FAISS vectorstore using similarity search."""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    index_path = os.path.join(vectorstore_path, "index.faiss")
    if not os.path.exists(index_path):
        print("FAISS index not found. Ensure vectorstore is created.")
        return

    index = faiss.read_index(index_path)

    query_embedding = np.array(embeddings.embed_query(query))
    query_embedding = query_embedding.reshape(1, -1)
    query_embedding = normalize_embeddings(query_embedding)

    distances, indices = index.search(query_embedding, k=10)

    print("Top results:")
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0 or idx >= len(documents):
            continue 
        
        doc = documents[idx]
        print(f"\nResult {i + 1}:")
        print(f"Title: {doc.metadata.get('title', 'No Title')}")
        print(f"Content Snippet: {doc.page_content[:300]}...")
        print(f"Source File: {doc.metadata.get('source_file', 'Unknown')}")
        print(f"Similarity Score: {1 - distance:.4f}")  
        print("-" * 50)

if __name__ == "__main__":
    print("Loading JSON files and creating documents...")
    documents = load_json_as_documents(json_folder)
    print(f"Loaded {len(documents)} documents.")

    index_path = os.path.join(vectorstore_path, "index.faiss")
    if not os.path.exists(index_path):
        print("Creating FAISS vectorstore...")
        create_faiss_vectorstore(documents, embedding_model, vectorstore_path)
        print("Vectorstore creation completed.")
    else:
        print("FAISS vectorstore already exists. Skipping creation.")

    test_faiss_vectorstore(vectorstore_path, embedding_model, query="Cardiovascular Risk Factors in PCOS")
