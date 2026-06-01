import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import argparse
import config
from fv_utils import FisherVectorExtractor
from dataset import ATRWDataset


# --- 可配置的消融模型定义 ---
class TigerMLP(nn.Module):
    def __init__(self, input_dim, num_classes, use_bn=True, dropout_rate=0.5):
        super(TigerMLP, self).__init__()
        layers = []

        # 第一层
        layers.append(nn.Linear(input_dim, 1024))
        if use_bn: layers.append(nn.BatchNorm1d(1024))
        layers.append(nn.ReLU())
        if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))

        # 第二层
        layers.append(nn.Linear(1024, 512))
        if use_bn: layers.append(nn.BatchNorm1d(512))
        layers.append(nn.ReLU())

        # 输出层
        layers.append(nn.Linear(512, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = f"K{args.kernels}_BN{args.bn}_Drop{args.dropout}"
    save_path = os.path.join(config.MODEL_SAVE_DIR, exp_name)
    if not os.path.exists(save_path): os.makedirs(save_path)

    # 1. 特征提取器初始化
    extractor = FisherVectorExtractor(n_kernels=args.kernels)
    gmm_file = os.path.join(save_path, 'gmm.pkl')

    # 2. 数据集准备
    train_dataset = ATRWDataset(mode='train', extractor=extractor, config=config, mapping_json=config.TRAIN)
    global_id2label = train_dataset.id2label

    # 保存该次实验的 ID 映射
    with open(os.path.join(save_path, 'id_mapping.json'), 'w') as f:
        json.dump(global_id2label, f)

    val_dataset = ATRWDataset(mode='val', extractor=extractor, config=config, mapping_json=config.VAL)
    val_dataset.id2label = global_id2label

    # 3. 如果没有 GMM 权重则训练 GMM
    if not os.path.exists(gmm_file):
        print(f">>> [{exp_name}] 正在训练 GMM...")
        train_imgs = [train_dataset.load_and_crop(s) for s in train_dataset.samples]
        extractor.fit(train_imgs)
        extractor.save_gmm(gmm_file)
    else:
        extractor.load_gmm(gmm_file)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 4. 初始化模型
    input_dim = 2 * config.SIFT_DIM * args.kernels
    model = TigerMLP(input_dim, train_dataset.num_classes, use_bn=args.bn, dropout_rate=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 5. 训练循环
    best_acc = 0.0
    for epoch in range(config.EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(save_path, 'mlp_best.pth'))
        print(f"Exp: {exp_name} | Epoch {epoch + 1} | Val Acc: {acc:.2f}%")

def str2bool(v):
    """将字符串转换为布尔值的辅助函数"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernels', type=int, default=64, help='GMM 聚类数')
    parser.add_argument('--bn', type=str2bool, default=True, help='是否使用 BatchNorm')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout 概率')
    args = parser.parse_args()
    run_experiment(args)