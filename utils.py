import torch
import json

from models import CLIPModel
import config as CFG


class Logger:
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, *msg):
        with open(self.filename, 'a') as f:
            print(*msg, file=f)


def parse_json_generator(path):
    with open(path) as f:
        for i, line in enumerate(f):
            yield json.loads(line)


def get_model(model_path):
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    return model