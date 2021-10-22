from transformers import AutoTokenizer
import itertools
import torch
import numpy as np
import json

from config import tqdm, logger
from models import CLIPModel
from inference import make_predictions
from .data_utils import get_transforms, build_loaders
import config as CFG


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def calc_accuracy(gt, pred, strict=True, average=False):
    assert set(gt.keys()) == set(pred.keys()), \
        "Some of the dataset keys are missing in the preditions: " + str((gt.keys()) ^ set(pred.keys()))

    results = dict()
    for dataset in gt.keys():
        if strict:
            assert set(gt[dataset].keys()) == set(pred[dataset].keys()), \
               "Some of the images are missing in the predictions: " + str(set(gt[dataset].keys()) ^ set(pred[dataset].keys()))
        keys = list(set(gt[dataset].keys()) & set(pred[dataset].keys()))
        pred_list = [pred[dataset][x] for x in keys]
        gt_list = [gt[dataset][x] for x in keys]
        accuracy = sum((x == y) for x, y in zip(pred_list, gt_list)) / len(pred_list)
        results[dataset] = accuracy * 100

    if average:
        acc = sum(results.values()) / len(results)
        return acc
    else:
        return results


def evaluate(model, tokenizer):
    gt = json.load(open(CFG.val_gt_path))
    pred = make_predictions(model, tokenizer)
    return calc_accuracy(gt, pred)


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    total = CFG.n_sample // CFG.batch_size * CFG.batch_size
    tqdm_obj = tqdm(train_loader, total=total, smoothing=0.8, leave=False)
    # tqdkm_obj = tqdm(train_loader, total=len(train_loader), smoothing=0.8, leave=False)
    for batch in train_loader:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k not in ('caption', 'orig_image')}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == "batch":
            lr_scheduler.step(loss.item())

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_obj.update(count)
        tqdm_obj.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    tqdm_obj.close()
    return loss_meter


def train():
    print(logger, 'is active')
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_tokenizer)
    _, train_loader = build_loaders(tokenizer, get_transforms(mode='train'), mode='train')

    # if pretrained:
    #     model = get_model(CFG.save_path)
    # else:
    #     model = CLIPModel().to(CFG.device)
    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_acc = 81.
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        valid_acc = evaluate(model, tokenizer)
        print(valid_acc)
        valid_acc_avg = np.mean(list(valid_acc.values()))
        if valid_acc_avg > best_acc:
            best_acc = valid_acc_avg
            torch.save(model.state_dict(), CFG.save_path)
            print("Saved Best Model!")
        lr_scheduler.step(valid_acc_avg)





