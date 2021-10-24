from PIL import Image
import requests
from io import BytesIO
import os
from multiprocessing import Queue

from config import logger
from config import cfg


def img_by_url(url, image_id, q):
    content = None
    try:
        content = requests.get(url).content
    except Exception as exc:
        logger(image_id, type(exc).__name__, url, exc)
    q.put((image_id, content, url))


def parse_images(imgs, q):
    for img_obj in imgs:
        img_by_url(img_obj['url'], img_obj['image'], q)


def write_images(q: Queue):
    while True:
        data = q.get()
        if data == 'kill':
            break
        image_id, img_content, url = data
        try:
            if img_content is not None:
                img = Image.open(BytesIO(img_content)).convert('RGB').resize((cfg.size,) * 2)
                img.save(os.path.join(cfg.images_path, str(image_id) + '.jpg'))
        except Exception as exc:
            logger(image_id, type(exc).__name__, url, exc)
