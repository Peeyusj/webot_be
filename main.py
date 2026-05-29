import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

print("Loading embedding model locally...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

# Expanded reference phrases to catch conversational language
SUMMARY_PHRASES = [
    "Summarize this page.",
    "What is the overarching theme?",
    "Give me the TL;DR.",
    "What is the main idea here?",
    "Can you give me the gist of this article?",
    "Write a master summary.",
    "Give me a quick breakdown of this whole thing.",
    "Explain the entire page.",
    "What is this website about?"
]

# Pre-compute the vectors for our summary reference space
SUMMARY_VECTORS = embedding_model.encode(SUMMARY_PHRASES)

class RouteRequest(BaseModel):
    question: str

# ==========================================
# PRODUCTION HEALTH CHECK ROUTE
# ==========================================
@app.get("/health")
def health_check():
    """Standard health check for monitoring and orchestrators."""
    return {"status": "healthy"}

# ==========================================
# INTENT ROUTER ROUTE
# ==========================================
@app.post("/api/route")
def determine_route(request: RouteRequest):
    question_vector = embedding_model.encode([request.question])[0]
    
    similarities = np.dot(SUMMARY_VECTORS, question_vector) / (
        np.linalg.norm(SUMMARY_VECTORS, axis=1) * np.linalg.norm(question_vector)
    )
    
    max_similarity = float(np.max(similarities))
    
    # Calibrated threshold down to 0.45 to catch conversational syntax
    intent = "GLOBAL" if max_similarity >= 0.45 else "SPECIFIC"
    
    return {
        "question": request.question,
        "highest_similarity": max_similarity,
        "intent": intent
    }