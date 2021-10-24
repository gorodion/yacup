from multiprocessing import Process, Queue
from more_itertools import chunked
import os

from download_utils import write_images
from config import tqdm
from config import cfg
from utils import parse_json_generator


if __name__ == '__main__':
    if cfg.debug:
        print('DEBUG mode')
    if cfg.with_async:
        print('With asynchronous')
        from async_utils import parse_images
    else:
        print('Without asynchronous')
        from download_utils import parse_images

    os.makedirs(cfg.images_path, exist_ok=True)
    existing_ids = {int(fname.split('.')[0]) for fname in os.listdir(cfg.images_path)}
    tqdm_obj = tqdm(
        total=cfg.total_images
            if cfg.total_images is not None
            else sum(1 for _ in parse_json_generator(cfg.images_path)),
        unit='img')
    save_q = Queue(maxsize=cfg.n_load)
    writer = Process(target=write_images, args=(save_q,))
    writer.start()

    try:
        for img_batch in chunked(parse_json_generator(cfg.image_urls_path), cfg.n_load):
            tqdm_obj.update(len(img_batch))
            img_batch = [im for im in img_batch
                         if im['image'] >= cfg.min_id
                         if im['image'] not in existing_ids]
            if img_batch:
                parse_images(img_batch, save_q)
            tqdm_obj.set_description(f'queue size: {save_q.qsize()}. saved images: {len(os.listdir(cfg.images_path))}')
    except KeyboardInterrupt:
        print('interrupted')
    finally:
        save_q.put('kill')
        save_q.close()
        save_q.join_thread()
        writer.join()
