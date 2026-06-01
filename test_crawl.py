import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from crawl4ai import AsyncWebCrawler

async def test():
    print("Starting crawler...")
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url="https://redux.js.org/introduction/getting-started")
        print(f"Success: {result.success}")
        print(f"Markdown length: {len(result.markdown or '')}")

# Don't use asyncio.run() — create loop manually
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    loop.run_until_complete(test())
finally:
    loop.close()