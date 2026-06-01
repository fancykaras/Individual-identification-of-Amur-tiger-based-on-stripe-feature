import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


# dataset.py 片段
class ATRWDataset(Dataset):
    def __init__(self, mode, extractor, config, mapping_json=None):
        self.mode = mode
        self.extractor = extractor
        self.config = config

        # 加载数据
        if mapping_json and os.path.exists(mapping_json):
            with open(mapping_json, 'r') as f:
                self.samples = json.load(f)
        else:
            self.samples = []

        # 默认构建映射表（会被 main.py 或 evaluate.py 的外部赋值覆盖）
        unique_ids = sorted(list(set([s['tiger_id'] for s in self.samples])))
        self.id2label = {tid: i for i, tid in enumerate(unique_ids)}
        self.num_classes = len(self.id2label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 1. 裁剪图片
        img = self.load_and_crop(sample)
        # 2. 提特征
        fv = self.extractor.fisher_vector(img)
        # 3. 取标签索引 (此处如果 id_mapping 没对齐会报 KeyError，已被 evaluate 的过滤逻辑解决)
        label = self.id2label[sample['tiger_id']]

        return torch.from_numpy(fv).float(), torch.tensor(label).long()

    def _load_bbox_map(self, json_path):
        """解析 keypoint 文件，获取每个文件名的 bbox"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        # print(data['images'])

        # 添加这一行来“排雷”
        # if len(data['images']) > 0:
        #     print(f">>> 调试信息：测试集 images 的第一个条目键名为: {data['images'][0].keys()}")

        # 建立 image_id -> file_name 映射
        if self.mode == 'train':
            img_id_to_name = {img['id']: img['file_name'] for img in data['images']}
        elif self.mode == 'test':
            img_id_to_name = {img['id']: img['filename'] for img in data['images']}
        # 建立 file_name -> bbox 映射
        bbox_map = {}
        for ann in data['annotations']:
            fname = img_id_to_name.get(ann['image_id'])
            if fname:
                # 只保留第一个发现的 bbox，或者根据需要处理一张图多只老虎的情况
                bbox_map[fname] = ann['bbox']
        return bbox_map

    def _parse_reid_data(self, json_path, split_key):
        """关联 split 里的 ID 和 BBox 映射表"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        target_ids = data[split_key][self.mode]
        entities = data['entities']

        samples = []
        for tid in target_ids:
            if tid in entities:
                for fname in entities[tid]:
                    # 只有当 keypoint 文件里有该图的 bbox 时才加入
                    if fname in self.bbox_map:
                        if self.mode == 'train':
                            samples.append({
                                'path': os.path.join(self.config.TRAIN_IMG_DIR, fname),
                                'bbox': self.bbox_map[fname],
                                'tiger_id': tid
                            })
                        elif self.mode == 'test':
                            samples.append({
                                'path': os.path.join(self.config.TEST_IMG_DIR, fname),
                                'bbox': self.bbox_map[fname],
                                'tiger_id': tid
                            })
        return samples

    def load_and_crop(self, item):
        # --- 修改前 ---
        # img = cv2.imread(item['path'])

        # --- 修改后 ---
        # 1. 从 item 中获取文件名（之前生成的 JSON 里叫 file_name）
        image_name = item.get('file_name')

        # 2. 结合 config 中定义的图片根目录，拼接成完整路径
        import os
        if self.mode == 'train':
            img_full_path = os.path.join(self.config.TRAIN_IMG_DIR, image_name)
        if self.mode == 'val':
            img_full_path = os.path.join(self.config.VAL_IMG_DIR, image_name)
        if self.mode == 'test':
            img_full_path = os.path.join(self.config.TEST_IMG_DIR, image_name)

        # 3. 读取图片
        img = cv2.imread(img_full_path)

        if img is None:
            raise FileNotFoundError(f"无法在路径加载图片: {img_full_path}")

        # 4. 获取 bbox 并裁剪
        # 确保 bbox 也在 JSON 中存在，我们生成的格式是 [x, y, w, h]
        x, y, w, h = map(int, item['bbox'])

        # 防止裁剪越界（安全处理）
        h_img, w_img = img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)

        img_crop = img[y1:y2, x1:x2]
        return img_crop

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_crop = self.load_and_crop(item)

        # 处理异常情况
        if img_crop is None:
            # 返回全零特征（维度需与 SIFT+FV 输出一致）
            fv_dim = 2 * self.config.SIFT_DIM * self.config.GMM_KERNELS
            return torch.zeros(fv_dim, dtype=torch.float32), 0

        # 提取 Fisher Vector
        try:
            fv = self.extractor.fisher_vector(img_crop)
        except:
            fv_dim = 2 * self.config.SIFT_DIM * self.config.GMM_KERNELS
            return torch.zeros(fv_dim, dtype=torch.float32), 0

        label = self.id2label[item['tiger_id']]
        return torch.from_numpy(fv).float(), label
