import torch
import albumentations as A

from .dataset import CLIPDataset
import config as CFG


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )


def build_loaders(tokenizer, transforms, mode="train"):
    dataset = CLIPDataset(tokenizer, transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        drop_last=True,
        # shuffle=True if mode == "train" else False
    )
    return dataset, dataloader
