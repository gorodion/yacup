from multiprocessing import Process, Queue
from more_itertools import chunked
import os

from download_tools.utils import write_images
from config import tqdm
import config as CFG
from utils import parse_json_generator


if __name__ == '__main__':
    if CFG.with_async:
        print('With asynchronous')
        from download_tools.async_utils import parse_images
    else:
        print('Without asynchronous')
        from download_tools.utils import parse_images

    os.makedirs(CFG.images_path, exist_ok=True)
    existing_ids = {int(fname.split('.')[0]) for fname in os.listdir(CFG.images_path)}
    tqdm_obj = tqdm(
        total=CFG.total_images
            if CFG.total_images is not None
            else sum(1 for _ in parse_json_generator(CFG.images_path)),
        unit='img')
    save_q = Queue(maxsize=CFG.n_load)
    writer = Process(target=write_images, args=(save_q,))
    writer.start()

    try:
        for img_batch in chunked(parse_json_generator(CFG.image_urls_path), CFG.n_load):
            tqdm_obj.update(len(img_batch))
            img_batch = [im for im in img_batch
                         if im['image'] >= CFG.min_id
                         if im['image'] not in existing_ids]
            if img_batch:
                parse_images(img_batch, save_q)
            tqdm_obj.set_description(f'queue size: {save_q.qsize()}. saved images: {len(os.listdir(CFG.images_path))}')
    except KeyboardInterrupt:
        print('interrupted')
    finally:
        save_q.put('kill')
        save_q.close()
        save_q.join_thread()
        writer.join()
