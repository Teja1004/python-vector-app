import json
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize Chroma client (persistent local DB)
client = chromadb.PersistentClient(path="./vector_store")
collection = client.get_or_create_collection("documents")

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load initial documents
def load_initial_docs(file_path="data/docs.json"):
    with open(file_path, "r") as f:
        docs = json.load(f)
    for doc in docs:
        embedding = model.encode([doc["text"]])[0].tolist()
        collection.add(
            ids=[doc["id"]],
            embeddings=[embedding],
            documents=[doc["text"]],
            metadatas=[{"category": doc["category"]}]
        )
    print("✅ Initial documents loaded into vector DB.")

# Add a new document
def add_document(doc_id: str, text: str, category: str):
    embedding = model.encode([text])[0].tolist()
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{"category": category}]
    )
    return {"status": "success", "id": doc_id}

# Query vector DB
def search_documents(query_text: str, top_k=3, category_filter=None):
    query_embedding = model.encode([query_text])[0].tolist()

    if category_filter:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"category": category_filter}
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

    matches = []
    docs_list = results['documents'][0] or []
    metadatas_list = results['metadatas'][0] or [None for _ in docs_list]

    for doc, metadata in zip(docs_list, metadatas_list):
        if metadata is None:
            category = "unknown"
        else:
            category = metadata.get("category", "unknown")
        matches.append({"text": doc, "category": category})

    return matches


