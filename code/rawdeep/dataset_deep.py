import torch
import cv2
import os
import json
from torch.utils.data import Dataset
from PIL import Image


class AtrwDeepDataset(Dataset):
    def __init__(self, mapping_json, img_dir, transform=None):
        with open(mapping_json, 'r') as f:
            self.samples = json.load(f)
        self.img_dir = img_dir
        self.transform = transform

        # 建立 ID 到索引的映射
        unique_ids = sorted(list(set([s['tiger_id'] for s in self.samples])))
        self.id2label = {tid: i for i, tid in enumerate(unique_ids)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.img_dir, item['file_name'])

        # 1. 加载图片
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            raise FileNotFoundError(f"无法读取: {img_path}")

        # 2. 转换颜色空间并根据 BBox 裁剪
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        x, y, w, h = map(int, item['bbox'])

        # 边界安全处理
        img_h, img_w = img_cv.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)

        crop = img_cv[y1:y2, x1:x2]

        # 3. 转换为 PIL 以适配 torchvision 的 transforms
        img_pil = Image.fromarray(crop)

        if self.transform:
            img_pil = self.transform(img_pil)

        label = self.id2label[item['tiger_id']]
        return img_pil, torch.tensor(label).long()