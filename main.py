import sys
import asyncio

# Windows fix: Playwright/Crawl4AI requires SelectorEventLoop
# FastAPI defaults to ProactorEventLoop on Windows which breaks subprocess spawning
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import os
import hashlib
import json
import asyncio
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import NoExtractionStrategy
import xml.etree.ElementTree as ET
import httpx

load_dotenv()

app = FastAPI()

# --- CORS (allows Chrome Extension to talk to localhost) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⚡ Starting WebChat Production Engine...")

# --- CHROMADB ---
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(name="webot_pages")
print("✅ ChromaDB connected!")

# --- GROQ CLIENT ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY missing from .env file!")

groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)
print("✅ Groq connected!")

# --- CONSTANTS ---
MAX_URLS = 50
MAX_WORDS_PER_PAGE = 5000
MAX_TOTAL_WORDS = 100000


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
    # Sent from frontend when user confirms full site indexing
    tab_id: int
    url: str           # current page URL — used to derive scope prefix
    confirmed: bool    # user clicked confirm on the page count dialog

class SiteDiscoverRequest(BaseModel):
    # First step — just discover pages without ingesting
    # Frontend calls this to show user the page count before confirming
    url: str

class ChatMessage(BaseModel):
    role: str
    content: str

class QueryData(BaseModel):
    tab_id: int
    url: str
    question: str
    history: List[ChatMessage] = []


# ============================================================
# UTILITIES
# ============================================================

def get_url_hash(url: str) -> str:
    """Truncated SHA-256 of URL for caching and filtering."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

def get_scope_prefix(url: str) -> str:
    """
    Derives crawl scope from current URL.
    
    Strategy:
    - 1 path segment  → use domain root (e.g. redux.js.org/)
    - 2 path segments → use first segment (e.g. nextjs.org/docs/)  
    - 3+ path segments → use first two segments (e.g. react.dev/reference/react/)
    
    Always strips trailing specifics to capture the whole section.
    """
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split("/") if p]

    if len(path_parts) == 0:
        # Already at root
        scope_parts = []
    elif len(path_parts) == 1:
        # e.g. /docs/ → use root
        scope_parts = []
    elif len(path_parts) == 2:
        # e.g. /introduction/getting-started → use /introduction/
        scope_parts = path_parts[:1]
    else:
        # e.g. /reference/react/useState → use /reference/react/
        scope_parts = path_parts[:2]

    scope_path = "/" + "/".join(scope_parts) + "/" if scope_parts else "/"

    scope_url = urlunparse((
        parsed.scheme,
        parsed.netloc,
        scope_path,
        "", "", ""
    ))
    
    print(f"🔍 Scope derived: {url} → {scope_url}")
    return scope_url

def get_domain_root(url: str) -> str:
    """Extract just the domain root for sitemap lookup."""
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))

def chunk_text_by_words(
    text: str,
    chunk_size: int = 150,
    overlap: int = 30
) -> List[str]:
    """Split text into overlapping word chunks."""
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
    """Hard trim text to a word limit."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

async def fetch_sitemap_urls(domain_root: str, scope_prefix: str) -> List[str]:
    """
    Attempts to fetch and parse sitemap.xml from the domain root.
    Filters URLs to only those matching the scope prefix.
    Returns empty list if sitemap not found or unparseable.
    """
    sitemap_url = domain_root.rstrip("/") + "/sitemap.xml"
    print(f"🗺️  Trying sitemap: {sitemap_url}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(sitemap_url)
            if response.status_code != 200:
                print(f"⚠️  Sitemap not found (HTTP {response.status_code})")
                return []
            
            # Parse the XML sitemap
            root = ET.fromstring(response.text)
            
            # Handle both standard sitemaps and sitemap index files
            namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            
            # Try sitemap index first (points to other sitemaps)
            sitemap_refs = root.findall(".//sm:sitemap/sm:loc", namespace)
            if sitemap_refs:
                # This is a sitemap index — fetch the first relevant child sitemap
                all_urls = []
                for ref in sitemap_refs[:5]:  # check first 5 child sitemaps
                    child_urls = await fetch_sitemap_urls(ref.text.strip(), scope_prefix)
                    all_urls.extend(child_urls)
                    if len(all_urls) >= MAX_URLS:
                        break
                return all_urls[:MAX_URLS]
            
            # Standard sitemap — extract all <loc> entries
            all_locs = root.findall(".//sm:url/sm:loc", namespace)
            urls = [loc.text.strip() for loc in all_locs if loc.text]
            
            # Filter to scope prefix
            scoped = [u for u in urls if u.startswith(scope_prefix)]
            print(f"✅ Sitemap found: {len(urls)} total, {len(scoped)} in scope")
            return scoped[:MAX_URLS]
            
    except Exception as e:
        print(f"⚠️  Sitemap fetch failed: {str(e)}")
        return []

async def crawl_urls_without_sitemap(
    scope_prefix: str,
    max_pages: int = MAX_URLS
) -> List[str]:
    """
    Fallback when no sitemap exists.
    Crawls the scope prefix URL and extracts internal links
    that stay within the same prefix scope.
    Does a shallow single-page link extraction first,
    then visits discovered pages up to the cap.
    """
    print(f"🔍 Crawling links under: {scope_prefix}")
    discovered = set()
    to_visit = [scope_prefix]

    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            while to_visit and len(discovered) < max_pages:
                url = to_visit.pop(0)

                if url in discovered:
                    continue

                try:
                    result = await crawler.arun(
                        url=url,
                        word_count_threshold=10,   # skip near-empty pages
                        bypass_cache=True,
                    )

                    if result.success:
                        discovered.add(url)

                        # Extract internal links staying within scope prefix
                        if result.links:
                            internal_links = result.links.get("internal", [])
                            for link_obj in internal_links:
                                # Crawl4AI returns links as dicts with 'href' key
                                href = link_obj.get("href", "") if isinstance(link_obj, dict) else str(link_obj)

                                # Normalise — strip fragments and query strings
                                href = href.split("#")[0].split("?")[0]

                                if (
                                    href
                                    and href.startswith(scope_prefix)
                                    and href not in discovered
                                    and href not in to_visit
                                ):
                                    to_visit.append(href)

                except Exception as page_err:
                    print(f"⚠️  Skipping {url}: {str(page_err)}")
                    continue

    except Exception as e:
        print(f"❌ Crawler init failed: {str(e)}")
        return []

    result_list = list(discovered)[:max_pages]
    print(f"✅ Discovered {len(result_list)} pages via link crawling")
    return result_list


# ============================================================
# GROQ HELPERS
# ============================================================

def route_question_intent(question: str) -> str:
    """
    Uses Groq to classify question intent into SPECIFIC / GLOBAL / HYBRID.
    Forced JSON output, max 20 tokens — near instant on Groq LPU.
    Falls back to SPECIFIC on any error.
    """
    system_prompt = """You are a search routing classifier.
Categorize the user's question into exactly ONE intent:

"SPECIFIC" - asking for a specific fact, code snippet, or isolated detail
"GLOBAL"   - asking for overview, summary, main themes, or general understanding  
"HYBRID"   - asking for comparisons, differences, or connections between multiple things

Respond ONLY with valid JSON: {"intent": "SPECIFIC"} or {"intent": "GLOBAL"} or {"intent": "HYBRID"}"""

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
        
        if intent not in ["SPECIFIC", "GLOBAL", "HYBRID"]:
            return "SPECIFIC"
        return intent
        
    except Exception as e:
        print(f"⚠️  Router fallback to SPECIFIC: {str(e)}")
        return "SPECIFIC"

def generate_page_summary(page_text: str, page_title: str) -> str:
    """
    Generates a concise 3-bullet summary of a single page.
    Used during full site ingestion for Master Summary pre-computation.
    Trims input to 3000 words to keep Groq calls fast.
    """
    trimmed = trim_to_word_limit(page_text, 3000)
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a documentation summarizer. "
                        "Summarize the provided page content into exactly 3 bullet points. "
                        "Each bullet should capture one core concept. Be concise."
                    )
                },
                {
                    "role": "user",
                    "content": f"Page: {page_title}\n\nContent:\n{trimmed}"
                }
            ],
            temperature=0.0,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️  Page summary failed for {page_title}: {str(e)}")
        return f"• Content from {page_title}"

def generate_master_summary(page_summaries: List[dict]) -> str:
    """
    Map-Reduce final step.
    Takes all per-page summaries and reduces them into one Master Summary.
    This is what gets returned for GLOBAL intent queries.
    
    page_summaries: [{"title": "...", "summary": "..."}, ...]
    """
    combined = "\n\n".join([
        f"### {item['title']}\n{item['summary']}"
        for item in page_summaries
    ])
    
    # Trim if somehow too long
    combined = trim_to_word_limit(combined, 8000)
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a technical documentation analyst. "
                        "You have been given summaries of every page in a documentation site. "
                        "Write a comprehensive Master Summary that covers: "
                        "1) The main purpose and scope of this documentation, "
                        "2) The key concepts and features covered, "
                        "3) How the different sections relate to each other. "
                        "Be thorough but structured. Use markdown headings."
                    )
                },
                {
                    "role": "user",
                    "content": f"Page summaries:\n\n{combined}"
                }
            ],
            temperature=0.0,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️  Master summary generation failed: {str(e)}")
        return "Master summary could not be generated."

def generate_single_page_summary(page_text: str, page_title: str) -> str:
    """
    Generates a summary for a single ingested page.
    Stored in ChromaDB with doc_type: 'page_summary'.
    Used for GLOBAL intent on single page mode.
    """
    trimmed = trim_to_word_limit(page_text, 5000)
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a documentation summarizer. "
                        "Write a comprehensive summary of this page covering all main points, "
                        "key concepts, and important details. Use markdown formatting."
                    )
                },
                {
                    "role": "user",
                    "content": f"Page: {page_title}\n\nContent:\n{trimmed}"
                }
            ],
            temperature=0.0,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️  Single page summary failed: {str(e)}")
        return "Summary could not be generated."


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health_check():
    return {"status": "healthy"}


# --- STEP 1 OF FULL SITE FLOW: DISCOVER PAGES ---
# Frontend calls this first to show user how many pages will be indexed
# before they confirm. No ingestion happens here.
@app.post("/discover-site")
async def discover_site(data: SiteDiscoverRequest):
    """
    Discovers how many pages exist under the scope prefix.
    Returns page count and scope so frontend can show confirmation dialog.
    Does NOT ingest anything.
    """
    try:
        scope_prefix = get_scope_prefix(data.url)
        domain_root = get_domain_root(data.url)
        
        print(f"🔍 Discovering pages under: {scope_prefix}")
        
        # Try sitemap first
        urls = await fetch_sitemap_urls(domain_root, scope_prefix)
        
        # Fallback to link crawling if no sitemap
        if not urls:
            # For discovery, just do a shallow check — don't recurse deeply
            urls = await crawl_urls_without_sitemap(scope_prefix, max_pages=MAX_URLS)
        
        page_count = len(urls)
        capped = page_count >= MAX_URLS
        
        # Check if already cached
        site_hash = get_url_hash(scope_prefix)
        existing = collection.get(
    where={"$and": [
        {"url_hash": {"$eq": site_hash}},
        {"doc_type": {"$eq": "master_summary"}}
    ]},
    limit=1
)
        already_cached = bool(existing and existing.get("ids"))
        
        return {
            "scope_prefix": scope_prefix,
            "page_count": page_count,
            "capped": capped,
            "cap_limit": MAX_URLS,
            "already_cached": already_cached,
            "urls_preview": urls[:5]  # first 5 for UI display
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- SINGLE PAGE INGESTION ---
@app.post("/ingest-page")
async def ingest_page_data(data: PageData):
    """
    Streams ingestion progress for a single page.
    1. Check cache
    2. Chunk text
    3. Embed and save to ChromaDB
    4. Generate page summary for GLOBAL queries
    """
    async def ingestion_generator():
        ingestion_start = time.time()
        
        try:
            url_hash = get_url_hash(data.url)
            
            yield f"data: {json.dumps({'status': 'processing', 'message': '🔍 Checking cache...'})}\n\n"
            await asyncio.sleep(0.1)
            
            # --- CACHE CHECK ---
            existing = collection.get(
                where={"url_hash": url_hash},
                limit=1
            )
            if existing and existing.get("ids") and len(existing["ids"]) > 0:
                elapsed = round(time.time() - ingestion_start, 2)
                yield f"data: {json.dumps({'status': 'ready', 'message': f'✨ Loaded from cache instantly! ({elapsed}s)', 'cached': True, 'elapsed': elapsed})}\n\n"
                return
            
            # --- CHUNKING ---
            yield f"data: {json.dumps({'status': 'processing', 'message': '✂️ Splitting text into chunks...'})}\n\n"
            
            # Apply per-page word cap
            trimmed_text = trim_to_word_limit(data.text, MAX_WORDS_PER_PAGE)
            text_chunks = chunk_text_by_words(trimmed_text, chunk_size=150, overlap=30)
            await asyncio.sleep(0.1)
            
            # --- EMBEDDING + SAVING ---
            yield f"data: {json.dumps({'status': 'processing', 'message': f'🧬 Embedding {len(text_chunks)} chunks...'})}\n\n"
            
            documents, metadatas, ids = [], [], []
            for idx, chunk in enumerate(text_chunks):
                documents.append(chunk)
                ids.append(f"url_{url_hash}_chunk_{idx}")
                metadatas.append({
                    "url_hash": url_hash,
                    "url": data.url,
                    "title": data.title,
                    "source": data.source,
                    "chunk_index": idx,
                    "doc_type": "chunk"
                })
            
            if documents:
                collection.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            # --- PAGE SUMMARY FOR GLOBAL QUERIES ---
            yield f"data: {json.dumps({'status': 'processing', 'message': '📝 Generating page summary...'})}\n\n"
            
            # Run summary generation in thread pool to avoid blocking async loop
            summary_text = await asyncio.get_event_loop().run_in_executor(
                None,
                generate_single_page_summary,
                data.text,
                data.title
            )
            
            # Save summary as a special document
            collection.upsert(
                documents=[summary_text],
                metadatas=[{
                    "url_hash": url_hash,
                    "url": data.url,
                    "title": data.title,
                    "doc_type": "page_summary"
                }],
                ids=[f"url_{url_hash}_page_summary"]
            )
            
            elapsed = round(time.time() - ingestion_start, 2)
            yield f"data: {json.dumps({'status': 'ready', 'message': f'✅ Indexed {len(text_chunks)} chunks in {elapsed}s', 'elapsed': elapsed, 'chunk_count': len(text_chunks)})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'status': 'failed', 'message': f'❌ Ingestion failed: {str(e)}'})}\n\n"
    
    return StreamingResponse(ingestion_generator(), media_type="text/event-stream")


# --- FULL SITE INGESTION ---
@app.post("/ingest-site")
async def ingest_site(data: SiteIngestRequest):
    """
    Full site ingestion pipeline with SSE progress streaming.
    
    Flow:
    1. Derive scope prefix from current URL
    2. Fetch URLs from sitemap (or crawl fallback)
    3. For each URL: crawl with Crawl4AI → chunk → embed → save
    4. Per-page summaries via Groq (Map phase)
    5. Master Summary from all page summaries (Reduce phase)
    6. Store Master Summary with doc_type: 'master_summary'
    """
    async def site_ingestion_generator():
        total_start = time.time()
        
        try:
            scope_prefix = get_scope_prefix(data.url)
            site_hash = get_url_hash(scope_prefix)
            domain_root = get_domain_root(data.url)
            
            yield f"data: {json.dumps({'status': 'processing', 'message': f'🔍 Checking cache for {scope_prefix}...'})}\n\n"
            await asyncio.sleep(0.1)
            
            # --- CACHE CHECK ---
            existing_summary = collection.get(
    where={"$and": [
        {"url_hash": {"$eq": site_hash}},
        {"doc_type": {"$eq": "master_summary"}}
    ]},
    limit=1
)
            if existing_summary and existing_summary.get("ids") and len(existing_summary["ids"]) > 0:
                elapsed = round(time.time() - total_start, 2)
                yield f"data: {json.dumps({'status': 'ready', 'message': f'✨ Site loaded from cache! ({elapsed}s)', 'cached': True, 'elapsed': elapsed})}\n\n"
                return
            
            # --- URL DISCOVERY ---
            yield f"data: {json.dumps({'status': 'processing', 'message': '🗺️ Discovering pages...'})}\n\n"
            
            urls = await fetch_sitemap_urls(domain_root, scope_prefix)
            
            if not urls:
                yield f"data: {json.dumps({'status': 'processing', 'message': '🔍 No sitemap found, crawling links...'})}\n\n"
                urls = await crawl_urls_without_sitemap(scope_prefix, max_pages=MAX_URLS)
            
            if not urls:
                yield f"data: {json.dumps({'status': 'failed', 'message': '❌ No pages found to index.'})}\n\n"
                return
            
            total_pages = len(urls)
            yield f"data: {json.dumps({'status': 'processing', 'message': f'📋 Found {total_pages} pages to index', 'page_count': total_pages})}\n\n"
            
            # --- CRAWL + CHUNK + EMBED EACH PAGE ---
            page_summaries = []     # collected for Master Summary later
            total_words = 0
            pages_indexed = 0
            
            async with AsyncWebCrawler(verbose=False) as crawler:
                for idx, page_url in enumerate(urls):
                    
                    # Check total word cap
                    if total_words >= MAX_TOTAL_WORDS:
                        yield f"data: {json.dumps({'status': 'processing', 'message': f'⚠️ Word limit reached at {pages_indexed} pages, finalizing...'})}\n\n"
                        break
                    
                    yield f"data: {json.dumps({'status': 'processing', 'message': f'📥 Scraping page {idx + 1}/{total_pages}: {page_url}', 'progress': idx + 1, 'total': total_pages})}\n\n"
                    
                    try:
                        # Crawl the page with Crawl4AI
                        result = await crawler.arun(url=page_url)
                        
                        if not result.markdown:
                            print(f"⚠️  Empty result for {page_url}, skipping")
                            continue
                        
                        # Apply per-page word cap
                        page_text = trim_to_word_limit(result.markdown, MAX_WORDS_PER_PAGE)
                        page_title = result.metadata.get("title", page_url) if result.metadata else page_url
                        
                        # Track total words
                        word_count = len(page_text.split())
                        total_words += word_count
                        
                        # Chunk the page
                        chunks = chunk_text_by_words(page_text, chunk_size=150, overlap=30)
                        
                        # Build documents for ChromaDB
                        # Note: we use site_hash (not page url_hash) so all pages
                        # are queryable together as one knowledge base
                        page_url_hash = get_url_hash(page_url)
                        documents, metadatas, ids = [], [], []
                        
                        for chunk_idx, chunk in enumerate(chunks):
                            documents.append(chunk)
                            ids.append(f"site_{site_hash}_page_{page_url_hash}_chunk_{chunk_idx}")
                            metadatas.append({
                                "url_hash": site_hash,        # site-level for querying
                                "page_url_hash": page_url_hash,
                                "url": page_url,
                                "title": page_title,
                                "source": "crawl4ai",
                                "chunk_index": chunk_idx,
                                "doc_type": "chunk"
                            })
                        
                        if documents:
                            collection.upsert(
                                documents=documents,
                                metadatas=metadatas,
                                ids=ids
                            )
                        
                        pages_indexed += 1
                        
                        # Generate per-page summary for Map phase
                        # Run in executor to avoid blocking the async loop
                        yield f"data: {json.dumps({'status': 'processing', 'message': f'📝 Summarizing page {idx + 1}/{total_pages}...'})}\n\n"
                        
                        page_summary = await asyncio.get_event_loop().run_in_executor(
                            None,
                            generate_page_summary,
                            page_text,
                            page_title
                        )
                        
                        page_summaries.append({
                            "title": page_title,
                            "url": page_url,
                            "summary": page_summary
                        })
                        
                    except Exception as page_error:
                        print(f"⚠️  Failed to process {page_url}: {str(page_error)}")
                        continue
            
            # --- MASTER SUMMARY (REDUCE PHASE) ---
            if page_summaries:
                yield f"data: {json.dumps({'status': 'processing', 'message': f'🧠 Generating Master Summary from {len(page_summaries)} pages...'})}\n\n"
                
                master_summary = await asyncio.get_event_loop().run_in_executor(
                    None,
                    generate_master_summary,
                    page_summaries
                )
                
                # Store Master Summary with special doc_type flag
                collection.upsert(
                    documents=[master_summary],
                    metadatas=[{
                        "url_hash": site_hash,
                        "url": scope_prefix,
                        "title": f"Master Summary: {scope_prefix}",
                        "doc_type": "master_summary",
                        "pages_indexed": pages_indexed
                    }],
                    ids=[f"site_{site_hash}_master_summary"]
                )
                
                print(f"✅ Master Summary saved for {scope_prefix}")
            
            # Final timing
            elapsed = round(time.time() - total_start, 2)
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
            
            yield f"data: {json.dumps({'status': 'ready', 'message': f'✅ Indexed {pages_indexed} pages in {time_str}', 'elapsed': elapsed, 'pages_indexed': pages_indexed, 'time_display': time_str})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'status': 'failed', 'message': f'❌ Site ingestion failed: {str(e)}'})}\n\n"
    
    return StreamingResponse(site_ingestion_generator(), media_type="text/event-stream")


# --- QUERY + ANSWER ---
@app.post("/ask-stream")
async def ask_question_stream(data: QueryData):
    """
    Full RAG query pipeline with intent routing.
    
    Flow:
    1. Route intent (SPECIFIC / GLOBAL / HYBRID)
    2. Execute appropriate retrieval strategy
    3. Stream sources payload
    4. Stream Groq answer tokens
    
    Timing is tracked and included in the sources payload.
    """
    try:
        query_start = time.time()
        url_hash = get_url_hash(data.url)
        
        # Detect if this is a site-level query or single page query
        # by checking if a master summary exists for this url_hash
        # If not, check scope prefix
        scope_prefix = get_scope_prefix(data.url)
        site_hash = get_url_hash(scope_prefix)
        
        # Check if full site was indexed
        
        site_indexed = collection.get(
    where={"$and": [
        {"url_hash": {"$eq": site_hash}},
        {"doc_type": {"$eq": "master_summary"}}
    ]},
    limit=1
)
        # Use site hash if full site was indexed, otherwise page hash
        active_hash = site_hash if (site_indexed and site_indexed.get("ids")) else url_hash
        
        print(f"🔍 Query mode: {'SITE' if active_hash == site_hash else 'PAGE'}")
        
        # --- INTENT ROUTING ---
        # Run in executor since it's a sync Groq call
        intent = await asyncio.get_event_loop().run_in_executor(
            None,
            route_question_intent,
            data.question
        )
        
        routing_time = round(time.time() - query_start, 2)
        print(f"🔮 Intent: {intent} (routed in {routing_time}s) — '{data.question[:50]}'")
        
        # --- RETRIEVAL STRATEGY ---
        search_results = None
        
        if intent == "GLOBAL":
            # Try to fetch pre-computed summary first

            summary_result = collection.get(
    where={"$and": [
        {"url_hash": {"$eq": active_hash}},
        {"doc_type": {"$eq": "master_summary"}}
    ]},
    limit=1
)
            
            # Fall back to page_summary for single page mode
            if not (summary_result and summary_result.get("ids")):
                summary_result = collection.get(
    where={"$and": [
        {"url_hash": {"$eq": url_hash}},
        {"doc_type": {"$eq": "page_summary"}}
    ]},
    limit=1
)
            
            if summary_result and summary_result.get("documents"):
                # Build a mock search_results structure from the summary
                search_results = {
                    "documents": [summary_result["documents"]],
                    "metadatas": [summary_result["metadatas"]]
                }
                print(f"✅ GLOBAL: Using pre-computed summary")
            else:
                # No summary found, fall back to vector search
                print(f"⚠️  GLOBAL: No summary found, falling back to vector search")
                search_results = collection.query(
                    query_texts=[data.question],
                    where={"url_hash": active_hash},
                    n_results=5
                )
        
        elif intent == "HYBRID":
            # Fetch more chunks for comparison questions
            search_results = collection.query(
                query_texts=[data.question],
                where={"url_hash": active_hash},
                n_results=5
            )
            print(f"✅ HYBRID: Fetched 5 chunks for comparison context")
        
        else:  # SPECIFIC
            search_results = collection.query(
                query_texts=[data.question],
                where={"url_hash": active_hash},
                n_results=3
            )
            print(f"✅ SPECIFIC: Fetched 3 chunks")
        
        retrieval_time = round(time.time() - query_start, 2)
        
        # --- VERIFY CONTEXT EXISTS ---
        if not search_results or not search_results.get("documents") or len(search_results["documents"][0]) == 0:
            async def fallback_generator():
                yield f"data: {json.dumps({'type': 'token', 'token': 'No indexed content found for this page. Please index it first.'})}\n\n"
            return StreamingResponse(fallback_generator(), media_type="text/event-stream")
        
        # --- COMPILE SOURCES ---
        retrieved_chunks = search_results["documents"][0]
        retrieved_metadata = search_results["metadatas"][0]
        
        sources_payload = []
        for idx, doc in enumerate(retrieved_chunks):
            meta = retrieved_metadata[idx]
            
            # Use page URL for clickable link
            # For full site mode each chunk has its own page URL
            source_url = meta.get("url", data.url)
            
            sources_payload.append({
                "index": idx + 1,
                "title": meta.get("title", "Source"),
                "snippet": doc[:150] + "..." if len(doc) > 150 else doc,
                "url": source_url,          # clickable link in frontend
                "doc_type": meta.get("doc_type", "chunk"),
                "intent_used": intent
            })
        
        retrieved_context = "\n---\n".join(retrieved_chunks)
        
        # --- BUILD SYSTEM PROMPT ---
        system_prompt = (
            "You are an intelligent AI assistant embedded in a browser extension.\n"
            "Answer the user's question using ONLY the provided context below.\n"
            "Be accurate, detailed, and well-structured. Use markdown formatting.\n"
            "Do not hallucinate or reference information outside the context.\n\n"
            f"--- CONTEXT ---\n{retrieved_context}\n--- END CONTEXT ---"
        )
        
        groq_messages = [{"role": "system", "content": system_prompt}]
        
        # Add sliding window history (last 6 messages = 3 exchanges)
        for msg in data.history[-6:]:
            groq_messages.append({"role": msg.role, "content": msg.content})
        
        groq_messages.append({"role": "user", "content": data.question})
        
        # --- STREAM RESPONSE ---
        def stream_tokens():
            generation_start = time.time()
            
            # Packet 1: Send sources + metadata before generation starts
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload, 'intent': intent, 'retrieval_time': retrieval_time})}\n\n"
            
            # Packet 2+: Stream Groq tokens
            response_stream = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=groq_messages,
                stream=True
            )
            
            for chunk in response_stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
            
            # Final packet: timing info
            generation_time = round(time.time() - generation_start, 2)
            total_time = round(time.time() - query_start, 2)
            
            yield f"data: {json.dumps({'type': 'done', 'generation_time': generation_time, 'total_time': total_time, 'time_display': f'{total_time}s'})}\n\n"
        
        return StreamingResponse(stream_tokens(), media_type="text/event-stream")
    
    except Exception as e:
        print(f"❌ [ASK STREAM ERROR]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))