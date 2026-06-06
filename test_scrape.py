import sys
import asyncio

# This MUST be set before creating any event loop
if sys.platform == "win32":
    # CORRECT — what Playwright actually needs
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from crawl4ai import AsyncWebCrawler

async def test():
    url = "https://redux.js.org/introduction/getting-started"
    
    print(f"Scraping: {url}")
    print("-" * 50)
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(
            url=url,
            page_timeout=15000,
        )
        
        if result.success:
            print(f"✅ Success")
            print(f"Title: {result.metadata.get('title', 'No title')}")
            print(f"Word count: {len(result.markdown.split())}")
            print(f"\nFirst 500 characters:")
            print("-" * 50)
            print(result.markdown[:500])
        else:
            print(f"❌ Failed: {result.error_message}")

# Use asyncio.run() NOT manual loop creation
# asyncio.run() respects the policy set above
asyncio.run(test())