from pathlib import Path
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as CFG
from train.dataset import ImageDataset # TODO dataset to separate class

def get_image_embeddings(model, image_dl):
    valid_image_embeddings = []
    with torch.no_grad():
        for imgs in image_dl:
            image_features = model.image_encoder(imgs.to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return torch.cat(valid_image_embeddings)

def get_text_embeddings(tokenizer, model, classes):
    encoded_classes = tokenizer(classes, padding=True, truncation=True, max_length=CFG.max_length)
    classes_dict = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_classes.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=classes_dict["input_ids"], attention_mask=classes_dict["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    return text_embeddings

def make_predictions(model, tokenizer):
    output = {}

    for dir in Path(CFG.val_data_path).iterdir():
        ds_name = dir.stem
        if ds_name == '.ipynb_checkpoints':
            continue
        output[ds_name] = {}

        img_paths = [str(i) for i in (dir.glob('img/*.jpg'))] # todo png
        img_names = [Path(path).stem for path in img_paths]
        with open(dir / 'classes.json') as f:
            classes = json.load(f)
        im_ds = ImageDataset(img_paths, mode='val')
        dl = DataLoader(im_ds, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=False)
        image_embeddings = get_image_embeddings(model, dl)
        text_embeddings = get_text_embeddings(tokenizer, model, classes)

        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        predicts = (image_embeddings @ text_embeddings.T).argmax(1).cpu().numpy()
        for img_name, predict in zip(img_names, predicts):
            output[ds_name][img_name] = int(predict)
    return output