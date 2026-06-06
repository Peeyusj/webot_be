import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
            print(f"\nFirst 500 characters of content:")
            print("-" * 50)
            print(result.markdown[:500])
        else:
            print(f"❌ Failed")

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    loop.run_until_complete(test())
finally:
    loop.close()