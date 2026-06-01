import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import os
import json
import config
from dataset_deep import AtrwDeepDataset  # 我们稍后定义的深度学习专用 Dataset


def train_deep():
    # 1. 环境配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 使用设备: {device}")

    if not os.path.exists(config.MODEL_SAVE_DIR):
        os.makedirs(config.MODEL_SAVE_DIR)

    # 2. 数据增强 (Deep Learning 的灵魂)
    # 训练集需要增强以防止过拟合
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 验证集只需 Resize 和归一化
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. 加载数据集
    # 注意：这里直接使用你之前生成的映射 JSON 文件
    train_dataset = AtrwDeepDataset(config.TRAIN, config.TRAIN_IMG_DIR, transform=train_transform)
    val_dataset = AtrwDeepDataset(config.VAL, config.VAL_IMG_DIR, transform=val_transform)

    # 强制使验证集的 ID 映射与训练集一致
    val_dataset.id2label = train_dataset.id2label

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.id2label)
    print(f">>> 数据加载完成，类别数: {num_classes}")

    # 4. 加载模型 (ResNet50)
    # 使用 ImageNet 预训练权重加速收敛
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # 修改最后的全连接层以匹配老虎的类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # 5. 定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    # 深度学习通常使用较小的学习率来微调预训练模型
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 保存 ID 映射供 evaluate 使用
    with open(os.path.join(config.MODEL_SAVE_DIR, 'deep_id_mapping.json'), 'w') as f:
        json.dump(train_dataset.id2label, f)

    # 6. 训练循环
    best_acc = 0.0
    epochs = 15  # 深度学习通常不需要太多 epoch 即可收敛

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 验证逻辑
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"--- Epoch {epoch + 1} 验证集准确率: {val_acc:.2f}% ---")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'resnet50_best.pth'))

    print(f">>> 训练结束，最佳准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    train_deep()