import os
import random
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
import torch
import albumentations as A

from config import logger
from utils import parse_json_generator
from config import cfg


class CLIPDataset(IterableDataset):
    def __init__(self, tokenizer, transforms):
        self.transforms = transforms
        self.tokenizer = tokenizer

    @staticmethod
    def make_choice_captions(captions):
        return [random.choice(caption['queries']) for caption in captions]
        # return [max(caption['queries'], key=len) for caption in captions]

    def iter_batch(self):
        images_ids = {int(i.split('.')[0]) for i in os.listdir('images')}
        captions = random.choices(
            [cap_obj for cap_obj in parse_json_generator(cfg.captions_path)
                        if cap_obj['image'] in images_ids],
            k=cfg.n_sample
        )
        encoded_captions = self.tokenizer(
            self.make_choice_captions(captions),
            padding=True,
            truncation=True,
            max_length=cfg.max_length
        )
        for i in range(len(captions)):
            image_id = captions[i]["image"]
            img = cv2.imread(f'images/{image_id}.jpg')
            if img is None:
                print(image_id, 'did not read')
                logger(image_id, 'did not read')
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            item = {
                key: torch.tensor(values[i])
                for key, values in encoded_captions.items()
            }

            if cfg.debug:
                item['orig_image'] = cv2.resize(img, (cfg.size, cfg.size))
                item['caption'] = captions[i]['queries'][0]
            img = self.transforms(image=img)['image']
            item['image'] = torch.tensor(img).permute(2, 0, 1).float()
            yield item

    def __iter__(self):
        for item in self.iter_batch():
            yield item


class ImageDataset(Dataset):
    def __init__(self, img_paths, mode='train'):
        self.img_paths = img_paths
        self.transforms = get_transforms(mode=mode)

    def __getitem__(self, i):
        img_path = self.img_paths[i]
        image = Image.open(img_path)
        image = np.array(image.convert('RGB'))
        image = self.transforms(image=image)['image']
        return torch.tensor(image).permute(2, 0, 1).float()

    def __len__(self):
        return len(self.img_paths)


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(cfg.size, cfg.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(cfg.size, cfg.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )


def build_loaders(tokenizer, transforms, mode="train"):
    dataset = CLIPDataset(tokenizer, transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        # shuffle=True if mode == "train" else False
    )
    return dataset, dataloader


if __name__ == '__main__':
    from transformers import AutoTokenizer
    dset, dload = build_loaders(AutoTokenizer.from_pretrained(cfg.text_tokenizer), get_transforms(mode='train'))
    print(next(iter(dload)))