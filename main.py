import hashlib
import json
import time
import asyncio  # Added for non-blocking UI transition delays
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import chromadb
import ollama

app = FastAPI()

print("⚡ Starting WebChat Stream Engine...")
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(name="webot_pages")
print("✅ Connected to ChromaDB!")

# --- DATA MODELS ---

class PageData(BaseModel):
    tab_id: int
    url: str
    title: str
    text: str
    source: str

class ChatMessage(BaseModel):
    role: str
    content: str

class QueryData(BaseModel):
    tab_id: int
    url: str
    question: str
    history: List[ChatMessage] = []

# --- UTILITIES ---

def get_url_hash(url: str) -> str:
    return hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]

def chunk_text_by_words(text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
        i += (chunk_size - overlap)
    return chunks

# --- STREAMING ENDPOINTS ---

@app.post("/ingest-page")
async def ingest_page_data(data: PageData):
    """Streams ingestion progress back to the client using SSE protocol blocks."""
    
    # FIX: Corrected lowercase naming to match the return statement exactly
    async def ingestion_generator():
        try:
            url_hash = get_url_hash(data.url)
            
            yield f"data: {json.dumps({'status': 'processing', 'message': '🔍 Checking database cache...'})}\n\n"
            await asyncio.sleep(0.2) # FIX: Non-blocking async sleep for smooth UI updates
            
            existing_records = collection.get(where={"url_hash": url_hash}, limit=1)
            
            if existing_records and existing_records.get("ids") and len(existing_records["ids"]) > 0:
                yield f"data: {json.dumps({'status': 'ready', 'message': '✨ Page loaded instantly from cache!', 'cached': True})}\n\n"
                return

            yield f"data: {json.dumps({'status': 'processing', 'message': '✂️ Parsing page text into distinct words...'})}\n\n"
            text_chunks = chunk_text_by_words(data.text, chunk_size=150, overlap=30)
            await asyncio.sleep(0.2)

            yield f"data: {json.dumps({'status': 'processing', 'message': f'🧬 Computing embeddings for {len(text_chunks)} text segments...'})}\n\n"
            
            documents, metadatas, ids = [], [], []
            for idx, chunk in enumerate(text_chunks):
                documents.append(chunk)
                ids.append(f"url_{url_hash}_chunk_{idx}")
                metadatas.append({
                    "url_hash": url_hash,
                    "url": data.url,
                    "title": data.title,
                    "source": data.source,
                    "chunk_index": idx
                })

            if documents:
                collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

            yield f"data: {json.dumps({'status': 'ready', 'message': f'💾 Context synchronized successfully! ({len(text_chunks)} chunks stored)'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'status': 'failed', 'message': f'❌ Ingestion failed: {str(e)}'})}\n\n"

    return StreamingResponse(ingestion_generator(), media_type="text/event-stream")


@app.post("/ask-stream")
async def ask_question_stream(data: QueryData):
    """Streams data packet references followed by raw generation tokens using SSE."""
    try:
        url_hash = get_url_hash(data.url)
        
        search_results = collection.query(
            query_texts=[data.question],
            where={"url_hash": url_hash}, 
            n_results=3  
        )
        
        if not search_results["documents"] or len(search_results["documents"][0]) == 0:
            async def fallback_generator():
                yield f"data: {json.dumps({'type': 'token', 'token': 'I don\'t have context for this page yet.'})}\n\n"
            return StreamingResponse(fallback_generator(), media_type="text/event-stream")

        retrieved_chunks = search_results["documents"][0]
        retrieved_metadata = search_results["metadatas"][0]
        
        sources_payload = []
        for idx, doc in enumerate(retrieved_chunks):
            meta = retrieved_metadata[idx]
            sources_payload.append({
                "index": idx + 1,
                "title": meta.get("title", "Unknown Source"),
                "snippet": doc[:120] + "...",
                "source_type": meta.get("source", "page")
            })

        retrieved_context = "\n---\n".join(retrieved_chunks)
        system_prompt = (
            "You are a helpful local AI assistant. Answer the user's question using ONLY the context below.\n"
            "Keep your responses concise and direct (under 3 sentences). Refer to your sources if needed.\n\n"
            f"--- CONTEXT ---\n{retrieved_context}\n---------------"
        )

        ollama_messages = [{"role": "system", "content": system_prompt}]
        for msg in data.history[-4:]:
            ollama_messages.append({"role": msg.role, "content": msg.content})
        ollama_messages.append({"role": "user", "content": data.question})

        def stream_tokens():
            # Packet 1: Send the structured source arrays right away
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload})}\n\n"
            
            # Packet 2+: Stream the actual wording characters
            response_stream = ollama.chat(model='llama3.2', messages=ollama_messages, stream=True)
            for chunk in response_stream:
                token = chunk.get('message', {}).get('content', '')
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        return StreamingResponse(stream_tokens(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))