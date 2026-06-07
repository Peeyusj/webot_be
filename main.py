import sys
import asyncio

# Windows fix: Playwright/Crawl4AI requires SelectorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    print(f"🔧 Event loop policy: {type(asyncio.get_event_loop_policy()).__name__}")

import os
import hashlib
import json
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
import httpx
from urllib.parse import urlparse, urljoin, urlunparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

load_dotenv()

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⚡ Starting WebChat Production Engine (True Single-Pass Streaming)...")

# --- CHROMADB ---
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(name="webot_pages")
print("✅ ChromaDB connected!")

# --- GROQ CLIENT ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY missing from .env file!")

# Using Llama 3.3 70B Versatile for strict RAG
groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)
print("✅ Groq connected!")

# --- CONSTANTS ---
SITEMAP_MAX_URLS = 40  # Broad cap if sitemap is cleanly provided
CRAWL_MAX_URLS = 25    # Tighter cap to prevent infinite loop crawling
MAX_WORDS_PER_PAGE = 5000
MAX_TOTAL_WORDS = 100000

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.svg', 
                    '.webp', '.ico', '.pdf', '.zip', '.mp4')


# ============================================================
# DATA MODELS
# ============================================================

class PageData(BaseModel):
    tab_id: int
    url: str
    title: str
    text: str
    source: str

class SiteIngestRequest(BaseModel):
    tab_id: int
    url: str

class ChatMessage(BaseModel):
    role: str
    content: str

class QueryData(BaseModel):
    tab_id: int
    url: str
    question: str
    history: List[ChatMessage] = []

class ClearCacheRequest(BaseModel):
    url: str
    clear_all: bool = False


# ============================================================
# UTILITIES
# ============================================================

def get_url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

def get_scope_prefix(url: str) -> str:
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split("/") if p]

    if len(path_parts) == 0 or len(path_parts) == 1:
        scope_parts = []
    elif len(path_parts) == 2:
        scope_parts = path_parts[:1]
    else:
        scope_parts = path_parts[:2]

    scope_path = "/" + "/".join(scope_parts) + "/" if scope_parts else "/"
    scope_url = urlunparse((parsed.scheme, parsed.netloc, scope_path, "", "", ""))
    return scope_url

def get_domain_root(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))

def chunk_text_by_words(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
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

def trim_to_word_limit(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

async def fetch_sitemap_urls(domain_root: str, scope_prefix: str) -> List[str]:
    """Fetches sitemap.xml and returns URLs matching the active sub-path scope."""
    sitemap_url = domain_root.rstrip("/") + "/sitemap.xml"
    print(f"🗺️  Trying sitemap: {sitemap_url}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(sitemap_url)
            if response.status_code != 200:
                return []
            
            root = ET.fromstring(response.text)
            namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            
            sitemap_refs = root.findall(".//sm:sitemap/sm:loc", namespace)
            if sitemap_refs:
                all_urls = []
                for ref in sitemap_refs[:5]:
                    child_urls = await fetch_sitemap_urls(ref.text.strip(), scope_prefix)
                    all_urls.extend(child_urls)
                    if len(all_urls) >= SITEMAP_MAX_URLS:
                        break
                return all_urls[:SITEMAP_MAX_URLS]
            
            all_locs = root.findall(".//sm:url/sm:loc", namespace)
            urls = [loc.text.strip() for loc in all_locs if loc.text]
            
            scoped = [
                u for u in urls 
                if u.startswith(scope_prefix) 
                and not any(u.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
            ]
            print(f"✅ Sitemap found: {len(scoped)} viable pages in scope")
            return scoped[:SITEMAP_MAX_URLS]
            
    except Exception as e:
        print(f"⚠️  Sitemap fetch failed: {str(e)}")
        return []


# ============================================================
# RAG ROUTING
# ============================================================

def route_question_intent(question: str) -> str:
    system_prompt = """Categorize the user's question into ONE intent:
"SPECIFIC" - explicit facts, code snippets, or precise details
"GLOBAL"   - summaries, overviews, main themes, or broad concepts
"HYBRID"   - comparisons, connections, or multi-part questions
"CHAT"     - simple greetings (hi, hello), thank yous, or casual pleasantries
Respond ONLY with valid JSON: {"intent": "SPECIFIC"}"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=20,
        )
        result = json.loads(response.choices[0].message.content)
        intent = result.get("intent", "SPECIFIC")
        return intent if intent in ["SPECIFIC", "GLOBAL", "HYBRID", "CHAT"] else "SPECIFIC"
    except:
        return "SPECIFIC"


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/clear-cache")
async def clear_cache(data: ClearCacheRequest):
    try:
        if data.clear_all:
            global collection
            db_client.delete_collection(name="webot_pages")
            collection = db_client.get_or_create_collection(name="webot_pages")
            return {"status": "success", "message": "All cached documents cleared."}
        else:
            url_hash = get_url_hash(data.url)
            scope_prefix = get_scope_prefix(data.url)
            site_hash = get_url_hash(scope_prefix)
            
            collection.delete(where={"url_hash": url_hash})
            collection.delete(where={"url_hash": site_hash})
            
            return {"status": "success", "message": f"Cache cleared for {data.url}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest-page")
async def ingest_page_data(data: PageData):
    async def ingestion_generator():
        ingestion_start = time.time()
        try:
            url_hash = get_url_hash(data.url)
            yield f"data: {json.dumps({'status': 'processing', 'message': '🔍 Checking cache...'})}\n\n"
            
            existing = collection.get(where={"url_hash": url_hash}, limit=1)
            if existing and existing.get("ids"):
                elapsed = round(time.time() - ingestion_start, 2)
                yield f"data: {json.dumps({'status': 'ready', 'message': f'✨ Loaded from cache!', 'cached': True, 'elapsed': elapsed})}\n\n"
                return
            
            yield f"data: {json.dumps({'status': 'processing', 'message': '✂️ Processing text...'})}\n\n"
            
            words = data.text.split()
            if len(words) < 100:
                yield f"data: {json.dumps({'status': 'failed', 'message': '⚠️ Page has too little text content to index.'})}\n\n"
                return

            trimmed_text = trim_to_word_limit(data.text, MAX_WORDS_PER_PAGE)
            text_chunks = chunk_text_by_words(trimmed_text)
            
            yield f"data: {json.dumps({'status': 'processing', 'message': f'🧬 Embedding {len(text_chunks)} chunks...'})}\n\n"
            
            documents, metadatas, ids = [], [], []
            for idx, chunk in enumerate(text_chunks):
                documents.append(chunk)
                ids.append(f"url_{url_hash}_chunk_{idx}")
                metadatas.append({
                    "url_hash": url_hash,
                    "url": data.url,
                    "title": data.title,
                    "doc_type": "chunk"
                })
            
            if documents:
                collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            
            elapsed = round(time.time() - ingestion_start, 2)
            yield f"data: {json.dumps({'status': 'ready', 'message': f'✅ Indexed completely', 'elapsed': elapsed})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'status': 'failed', 'message': f'❌ Ingestion failed: {str(e)}'})}\n\n"
    
    return StreamingResponse(ingestion_generator(), media_type="text/event-stream")

@app.post("/ingest-site")
async def ingest_site(data: SiteIngestRequest):
    async def site_ingestion_generator():
        total_start = time.time()
        try:
            scope_prefix = get_scope_prefix(data.url)
            site_hash = get_url_hash(scope_prefix)
            domain_root = get_domain_root(data.url)
            
            # --- 1. CACHE CHECK ---
            yield f"data: {json.dumps({'status': 'processing', 'message': '🔍 Checking memory vault...'})}\n\n"
            
            existing = collection.get(where={"url_hash": site_hash}, limit=1)
            if existing and existing.get("ids"):
                elapsed = round(time.time() - total_start, 2)
                yield f"data: {json.dumps({'status': 'ready', 'message': '✨ Knowledge base loaded from memory!', 'cached': True, 'elapsed': elapsed})}\n\n"
                return
            
            yield f"data: {json.dumps({'status': 'processing', 'message': '🗺️ Looking for a sitemap... This might take a minute.'})}\n\n"
            sitemap_urls = await fetch_sitemap_urls(domain_root, scope_prefix)
            
            total_words = 0
            pages_indexed = 0
            
            async with AsyncWebCrawler(verbose=False) as crawler:
                run_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS, 
                    exclude_external_links=True
                )
                
                # ============================================================
                # SITEMAP FLOW: URLs are known, so we just scrape and index
                # ============================================================
                if sitemap_urls:
                    yield f"data: {json.dumps({'status': 'processing', 'message': f'✅ Valid sitemap found! Preparing to index up to {SITEMAP_MAX_URLS} pages.'})}\n\n"
                    total_pages = len(sitemap_urls)
                    
                    for idx, page_url in enumerate(sitemap_urls):
                        if total_words >= MAX_TOTAL_WORDS:
                            break
                        
                        yield f"data: {json.dumps({'status': 'processing', 'message': f'📥 Reading & Parsing document {idx + 1} of {total_pages}...', 'progress': idx + 1, 'total': total_pages})}\n\n"
                        
                        try:
                            result = await crawler.arun(url=page_url, config=run_config)
                            if not result.markdown:
                                continue
                            
                            word_count_check = len(result.markdown.split())
                            if word_count_check < 200:
                                continue
                            
                            page_text = trim_to_word_limit(result.markdown, MAX_WORDS_PER_PAGE)
                            page_title = result.metadata.get("title", "Documentation Page") if result.metadata else "Documentation Page"
                            
                            total_words += word_count_check
                            chunks = chunk_text_by_words(page_text)
                            
                            page_url_hash = get_url_hash(page_url)
                            documents, metadatas, ids = [], [], []
                            
                            for chunk_idx, chunk in enumerate(chunks):
                                documents.append(chunk)
                                ids.append(f"site_{site_hash}_page_{page_url_hash}_chunk_{chunk_idx}")
                                metadatas.append({
                                    "url_hash": site_hash,
                                    "url": page_url,
                                    "title": page_title,
                                    "doc_type": "chunk"
                                })
                            
                            if documents:
                                collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
                            
                            pages_indexed += 1
                            await asyncio.sleep(0.5)
                            
                        except Exception as page_error:
                            print(f"⚠️  Failed to parse {page_url}: {str(page_error)}")
                            continue
                
                # ============================================================
                # BFS FALLBACK FLOW: Discover, Scrape, and Index in ONE pass
                # ============================================================
                else:
                    yield f"data: {json.dumps({'status': 'processing', 'message': f'🔍 No sitemap found. Beginning unified deep-crawl & indexing...'})}\n\n"
                    
                    visited = set()
                    to_visit = [scope_prefix]
                    
                    while to_visit and pages_indexed < CRAWL_MAX_URLS:
                        if total_words >= MAX_TOTAL_WORDS:
                            break
                            
                        current_url = to_visit.pop(0)
                        if current_url in visited:
                            continue
                        visited.add(current_url)
                        
                        yield f"data: {json.dumps({'status': 'processing', 'message': f'📥 Exploring & Indexing page {pages_indexed + 1}/{CRAWL_MAX_URLS}...', 'progress': pages_indexed + 1, 'total': CRAWL_MAX_URLS})}\n\n"
                        
                        try:
                            # SCRAPE ONCE
                            result = await crawler.arun(url=current_url, config=run_config)
                            if not result.success:
                                continue
                            
                            # STEP A: Extract Links to keep exploring
                            internal_links = result.links.get("internal", []) if result.links else []
                            for link_item in internal_links:
                                href = link_item.get("href", "")
                                if not href:
                                    continue
                                
                                full_url = urljoin(result.url, href)
                                parsed = urlparse(full_url)
                                clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
                                
                                if clean_url.startswith(scope_prefix) and clean_url not in visited and clean_url not in to_visit:
                                    if not any(clean_url.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                                        to_visit.append(clean_url)
                            
                            # STEP B: Index the Markdown from that exact same scrape
                            if not result.markdown:
                                continue
                                
                            word_count_check = len(result.markdown.split())
                            if word_count_check < 200:
                                continue # Skip indexing it, but we successfully kept its links!
                                
                            page_text = trim_to_word_limit(result.markdown, MAX_WORDS_PER_PAGE)
                            page_title = result.metadata.get("title", "Documentation Page") if result.metadata else "Documentation Page"
                            
                            total_words += word_count_check
                            chunks = chunk_text_by_words(page_text)
                            
                            page_url_hash = get_url_hash(current_url)
                            documents, metadatas, ids = [], [], []
                            
                            for chunk_idx, chunk in enumerate(chunks):
                                documents.append(chunk)
                                ids.append(f"site_{site_hash}_page_{page_url_hash}_chunk_{chunk_idx}")
                                metadatas.append({
                                    "url_hash": site_hash,
                                    "url": current_url,
                                    "title": page_title,
                                    "doc_type": "chunk"
                                })
                            
                            if documents:
                                collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
                            
                            pages_indexed += 1
                            await asyncio.sleep(0.5)
                            
                        except Exception as e:
                            print(f"⚠️  Failed to process {current_url}: {e}")
                            continue

            # Finish Response
            elapsed = round(time.time() - total_start, 2)
            
            if pages_indexed == 0:
                yield f"data: {json.dumps({'status': 'failed', 'message': '❌ Failed to read valid text from the discovered pages.'})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'ready', 'message': f'✅ Successfully mapped {pages_indexed} pages into memory.', 'elapsed': elapsed})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'status': 'failed', 'message': f'❌ System error during ingestion: {str(e)}'})}\n\n"
    
    return StreamingResponse(site_ingestion_generator(), media_type="text/event-stream")

@app.post("/ask-stream")
async def ask_question_stream(data: QueryData):
    try:
        url_hash = get_url_hash(data.url)
        scope_prefix = get_scope_prefix(data.url)
        site_hash = get_url_hash(scope_prefix)
        
        site_indexed = collection.get(where={"url_hash": site_hash}, limit=1)
        active_hash = site_hash if (site_indexed and site_indexed.get("ids")) else url_hash
        
        intent = await asyncio.get_event_loop().run_in_executor(None, route_question_intent, data.question)
        
        # --- CASUAL CHAT BYPASS ---
        if intent == "CHAT":
            async def chat_generator():
                yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
                response_stream = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a friendly AI assistant. Respond warmly and concisely to the user's greeting or pleasantry."},
                        {"role": "user", "content": data.question}
                    ],
                    stream=True,
                    temperature=0.5
                )
                for chunk in response_stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return StreamingResponse(chat_generator(), media_type="text/event-stream")
        
        n_chunks = 8 if intent in ["GLOBAL", "HYBRID"] else 4
        
        search_results = collection.query(
            query_texts=[data.question],
            where={"url_hash": active_hash},
            n_results=n_chunks
        )
        
        if not search_results or not search_results.get("documents") or len(search_results["documents"][0]) == 0:
            async def fallback_generator():
                yield f"data: {json.dumps({'type': 'token', 'token': 'No information found about this in documents.'})}\n\n"
            return StreamingResponse(fallback_generator(), media_type="text/event-stream")
        
        retrieved_chunks = search_results["documents"][0]
        retrieved_metadata = search_results["metadatas"][0]
        
        sources_payload = []
        seen_titles = set()
        
        for idx, meta in enumerate(retrieved_metadata):
            title = meta.get("title", "Indexed Document").strip()
            title = title.split("|")[0].split("-")[0].strip() 
            
            if title not in seen_titles and len(title) > 3:
                seen_titles.add(title)
                sources_payload.append({
                    "id": idx + 1,
                    "title": title
                })
        
        retrieved_context = "\n---\n".join(retrieved_chunks)
        
        system_prompt = (
            "You are an intelligent, strictly factual AI assistant.\n"
            "You must answer the user's question using ONLY the provided context below.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. If the answer cannot be fully and explicitly determined from the context, you MUST state exactly: "
            "\"No information found about this in documents.\"\n"
            "2. Do not attempt to guess, infer, or use outside knowledge under any circumstances.\n"
            "3. If the context contains the answer, be accurate, detailed, and well-structured using markdown.\n\n"
            f"--- CONTEXT ---\n{retrieved_context}\n--- END CONTEXT ---"
        )
        
        groq_messages = [{"role": "system", "content": system_prompt}]
        for msg in data.history[-4:]:
            groq_messages.append({"role": msg.role, "content": msg.content})
        groq_messages.append({"role": "user", "content": data.question})
        
        def stream_tokens():
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload})}\n\n"
            
            response_stream = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=groq_messages,
                stream=True,
                temperature=0.0
            )
            
            for chunk in response_stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return StreamingResponse(stream_tokens(), media_type="text/event-stream")
    
    except Exception as e:
        print(f"❌ [ASK STREAM ERROR]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
