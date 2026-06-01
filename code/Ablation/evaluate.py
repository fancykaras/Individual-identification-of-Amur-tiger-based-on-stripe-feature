import torch
import os
import json
import numpy as np
import config
from fv_utils import FisherVectorExtractor
from dataset import ATRWDataset
from main import TigerMLP  # 确保从 main.py 导入模型类

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize


def evaluate_ablation(exp_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 动态解析实验参数
    # 假设文件夹名格式为: K64_BNTrue_Drop0.5
    parts = exp_folder.split('_')
    n_kernels = int(parts[0][1:])
    use_bn = "True" in parts[1]
    dropout = float(parts[2][4:])

    work_dir = os.path.join(config.MODEL_SAVE_DIR, exp_folder)

    # 1. 加载映射和提取器
    with open(os.path.join(work_dir, 'id_mapping.json'), 'r') as f:
        id2label = json.load(f)
    num_classes = len(id2label)
    label2id = {int(v): k for k, v in id2label.items()}

    extractor = FisherVectorExtractor(n_kernels=n_kernels)
    extractor.load_gmm(os.path.join(work_dir, 'gmm.pkl'))

    # 2. 初始化模型
    input_dim = 2 * config.SIFT_DIM * n_kernels
    model = TigerMLP(input_dim, num_classes, use_bn=use_bn, dropout_rate=dropout).to(device)
    model.load_state_dict(torch.load(os.path.join(work_dir, 'mlp_best.pth'), map_location=device))
    model.eval()

    # 3. 数据加载
    test_dataset = ATRWDataset(mode='test', extractor=extractor, config=config, mapping_json=config.TEST)
    test_dataset.samples = [s for s in test_dataset.samples if s['tiger_id'] in id2label]
    test_dataset.id2label = id2label

    # --- 统计变量 ---
    all_true, all_pred, all_scores = [], [], []
    top1_c, top5_c, total = 0, 0, 0

    print(f">>> 正在评估实验: {exp_folder}")
    with torch.no_grad():
        for i in range(len(test_dataset)):
            feat, label = test_dataset[i]
            feat_tensor = feat.unsqueeze(0).to(device)
            outputs = model(feat_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # Top-K 计算
            maxk = min(5, num_classes)
            _, topk_idx = outputs.topk(maxk, 1)
            topk_list = topk_idx.cpu().numpy().flatten().tolist()

            total += 1
            all_true.append(label)
            all_pred.append(topk_list[0])
            all_scores.append(probs.cpu().numpy()[0])

            if label == topk_list[0]: top1_c += 1
            if label in topk_list: top5_c += 1

    # 4. 指标计算
    top1_acc = top1_c / total
    top5_acc = top5_c / total
    y_true_bin = label_binarize(all_true, classes=list(range(num_classes)))
    mAP = average_precision_score(y_true_bin, np.array(all_scores), average="macro")

    print(f"\n[结果 - {exp_folder}]")
    print(f"Top-1: {top1_acc:.2%} | Top-5: {top5_acc:.2%} | mAP: {mAP:.4f}\n")

    # 5. 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(all_true, all_pred, labels=list(range(num_classes)))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title(f'Confusion Matrix: {exp_folder}')
    plt.savefig(os.path.join(work_dir, 'confusion_matrix.png'))
    plt.close()


if __name__ == '__main__':
    # 评估基准实验
    # evaluate_ablation("K64_BNTrue_Drop0.5")

    # 评估增强实验
    evaluate_ablation("K64_BNTrue_Drop0.0")