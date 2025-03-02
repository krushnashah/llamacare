import os
import json
import numpy as np
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# File paths
json_folder = "articles"
vectorstore_path = "vectorstore/chromadb"
embedding_model = "sentence-transformers/all-mpnet-base-v2"

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=vectorstore_path)

def preprocess_text(text):
    """Cleans and normalizes text for improved similarity search."""
    text = text.lower().strip()
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

                tables_content = " ".join([table.get('content', '') for table in data.get("tables", [])])
                figures_content = " ".join([figure.get('caption', '') for figure in data.get("figures", [])])

                content = f"""
                    Title: {data.get('title', 'No Title')}
                    Abstract: {data.get('abstract', 'No Abstract')}
                    Body: {data.get('body', 'No Body')}
                    Tables: {tables_content}
                    Figures: {figures_content}
                """.strip()

                content = preprocess_text(content)
                
                if content:
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

def create_chromadb_vectorstore(documents, embedding_model):
    """Creates and stores document embeddings in ChromaDB."""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    collection = chroma_client.get_or_create_collection(name="pcos_articles", metadata={"hnsw:space": "cosine"})

    document_texts = [doc.page_content for doc in documents]
    document_embeddings = np.array(embeddings.embed_documents(document_texts))

    # Normalize embeddings
    document_embeddings /= np.linalg.norm(document_embeddings, axis=1, keepdims=True)

    existing_docs = collection.count()
    if existing_docs > 0:
        print(f"ChromaDB already contains {existing_docs} documents. Skipping duplicate storage.")
        return

    for i, doc in enumerate(documents):
        collection.add(
            ids=[str(i)],
            embeddings=[document_embeddings[i]],
            metadatas=[doc.metadata],
            documents=[doc.page_content],
        )

    print(f"ChromaDB now contains {collection.count()} documents.")

def test_chromadb_vectorstore(embedding_model, query="Test query"):
    """Tests ChromaDB by performing a similarity search."""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    try:
        collection = chroma_client.get_collection(name="pcos_articles")
    except chromadb.errors.InvalidCollectionException:
        print("Collection 'pcos_articles' not found. Creating it now...")
        collection = chroma_client.get_or_create_collection(name="pcos_articles")

    stored_count = collection.count()
    print(f"Stored Documents Count: {stored_count}")

    if stored_count == 0:
        print("No stored documents found in ChromaDB! Try recreating the vectorstore.")
        return

    query_embedding = np.array(embeddings.embed_query(query))
    query_embedding /= np.linalg.norm(query_embedding)

    results = collection.query(query_embeddings=[query_embedding], n_results=20)

    if not results["documents"][0]:
        print("No matching results found in ChromaDB!")
        return

    print("Top results:")
    for i, doc in enumerate(results["documents"][0]):
        metadata = results['metadatas'][0][i]
        print(f"\nResult {i + 1}:")
        print(f"Title: {metadata.get('title', 'No Title')}")
        print(f"Content Snippet: {doc[:300]}...")
        print(f"Source File: {metadata.get('source_file', 'Unknown')}")
        print(f"Similarity Score: {1 - results['distances'][0][i]:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    print("Loading JSON files and creating documents...")
    documents = load_json_as_documents(json_folder)
    print(f"Loaded {len(documents)} documents.")

    try:
        collection = chroma_client.get_collection(name="pcos_articles")
        print("ChromaDB collection already exists. Skipping creation.")
    except chromadb.errors.InvalidCollectionException:
        print("Creating ChromaDB collection...")
        create_chromadb_vectorstore(documents, embedding_model)
        print("ChromaDB collection creation completed.")

    test_chromadb_vectorstore(embedding_model, query="Cardiovascular Risk Factors in PCOS")
