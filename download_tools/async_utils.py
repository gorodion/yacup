import aiohttp
import asyncio

from config import logger
import config as CFG


async def img_by_url(url, image_id, session, q, sem):
    content = None
    async with sem:
        try:
            # timeout = aiohttp.ClientTimeout(total=30)
            async with session.get(url) as resp:
                content = await resp.read()
        except asyncio.TimeoutError as exc:
            logger(image_id, type(exc).__name__, url, exc)
        except asyncio.CancelledError:
            print('cancelled')
        except Exception as exc:
            logger(image_id, type(exc).__name__, url, exc)
        q.put((image_id, content, url))


async def parse_images_async(imgs, q):
    # timeout = aiohttp.ClientTimeout(total=None)
    sem = asyncio.Semaphore(CFG.sem)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for img_obj in imgs:
            task = asyncio.create_task(img_by_url(img_obj['url'], img_obj['image'], session, q, sem))
            tasks.append(task)
        await asyncio.gather(*tasks)


def parse_images(imgs, q):
    asyncio.run(parse_images_async(imgs, q))
    # asyncio.get_event_loop().run_until_complete(parse_images_async(imgs, q))
