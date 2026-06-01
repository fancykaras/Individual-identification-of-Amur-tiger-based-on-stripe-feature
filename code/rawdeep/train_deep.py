import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from dataset_deep import AtrwDeepDataset
import config


def train_deep_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 定义数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 加载数据
    train_set = AtrwDeepDataset('mapped_train.json', config.TRAIN_IMG_DIR, train_transform)
    val_set = AtrwDeepDataset('mapped_val.json', config.VAL_IMG_DIR, train_transform)  # 验证集通常不用复杂增强

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # 3. 构建模型 (ResNet50)
    num_classes = len(train_set.id2label)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 修改输出层
    model = model.to(device)

    # 4. 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 5. 训练循环
    for epoch in range(15):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证部分（省略部分代码，逻辑同前文）
        print(f"Epoch {epoch + 1} 完成训练")

    torch.save(model.state_dict(), "resnet50_tiger.pth")