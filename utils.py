import torch
import json


class Logger:
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, *msg):
        with open(self.filename, 'a') as f:
            print(*msg, file=f)


def parse_json_generator(path):
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            yield json.loads(line)
