import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import numpy as np
import config
from fv_utils import FisherVectorExtractor
from dataset import ATRWDataset


# --- 模型定义 ---
class TigerMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TigerMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def train():
    # 1. 环境配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(config.MODEL_SAVE_DIR):
        os.makedirs(config.MODEL_SAVE_DIR)

    # 2. 初始化特征提取器 (GMM)
    extractor = FisherVectorExtractor(n_kernels=config.GMM_KERNELS)
    gmm_path = os.path.join(config.MODEL_SAVE_DIR, 'gmm.pkl')
    if os.path.exists(gmm_path):
        print(">>> 加载预训练 GMM...")
        extractor.load_gmm(gmm_path)

    # 3. 加载数据集
    # 训练集：用于构建全局 ID 映射
    print(">>> 正在从 mapped_train.json 加载训练数据...")
    train_dataset = ATRWDataset(
        mode='train',
        extractor=extractor,
        config=config,
        mapping_json=config.TRAIN
    )

    # 核心：保存训练集的 ID 映射表，确保验证集和评估脚本使用相同的逻辑
    global_id2label = train_dataset.id2label
    mapping_save_path = os.path.join(config.MODEL_SAVE_DIR, 'id_mapping.json')
    with open(mapping_save_path, 'w') as f:
        json.dump(global_id2label, f)

    # 验证集：强制使用训练集的映射表
    print(">>> 正在从 mapped_val.json 加载验证数据...")
    val_dataset = ATRWDataset(
        mode='val',
        extractor=extractor,
        config=config,
        mapping_json=config.VAL
    )
    val_dataset.id2label = global_id2label  # 覆盖验证集的映射，保持索引一致

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f">>> 训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")
    print(f">>> 类别总数 (num_classes): {train_dataset.num_classes}")

    # 4. GMM 训练逻辑
    if not os.path.exists(gmm_path):
        print(">>> 正在训练 GMM...")
        train_imgs = [train_dataset.load_and_crop(s) for s in train_dataset.samples]
        extractor.fit(train_imgs)
        extractor.save_gmm(gmm_path)

    # 5. 初始化模型
    input_dim = 2 * config.SIFT_DIM * config.GMM_KERNELS
    model = TigerMLP(input_dim, train_dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)

    # 6. 训练循环
    best_acc = 0.0
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{config.EPOCHS}] Loss: {running_loss / len(train_loader):.4f} Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'mlp_final.pth'))

    print(f">>> 训练结束，最佳验证集准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    train()