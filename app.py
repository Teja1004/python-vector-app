from fastapi import FastAPI
from pydantic import BaseModel
from vector_db import load_initial_docs, add_document, search_documents

app = FastAPI(title="Vector DB Semantic Search API")

# Load initial documents
load_initial_docs()

# Pydantic models
class Document(BaseModel):
    id: str
    text: str
    category: str

class Query(BaseModel):
    query: str
    top_k: int = 3
    category: str = None

# Add new document
@app.post("/add")
def add_doc(doc: Document):
    return add_document(doc.id, doc.text, doc.category)

# Search documents
@app.post("/search")
def search_docs(query: Query):
    results = search_documents(query.query, top_k=query.top_k, category_filter=query.category)
    return {"query": query.query, "results": results}
