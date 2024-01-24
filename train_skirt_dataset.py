import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./new_data/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        with open('./new_data/top_prompt.json', 'rt') as f1:
            for line in f1:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./new_data/' + source_filename)
        target = cv2.imread('./new_data/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source = cv2.resize(source, (512,512))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target, (512,512))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


dataset = MyDataset()
# print(dataset['source'])
# for step, batch in enumerate(dataset):
#     print(batch["jpg"])
item = dataset[1234]
# jpg = item['jpg']
# txt = item['txt']
# hint = item['hint']
# print(item['jpg'])
# print(jpg.shape)
# print(hint.shape)
# print(pixel_values)

