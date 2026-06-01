import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import json
import cv2
from PIL import Image
import config
import numpy as np

# 新增用于指标计算和绘图的库
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize


def evaluate_deep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载 ID 映射表
    mapping_path = os.path.join(config.MODEL_SAVE_DIR, 'deep_id_mapping.json')
    if not os.path.exists(mapping_path):
        raise FileNotFoundError("未找到 deep_id_mapping.json")

    with open(mapping_path, 'r') as f:
        id2label = json.load(f)

    label2id = {int(v): k for k, v in id2label.items()}
    num_classes = len(id2label)

    # 2. 预处理
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. 加载模型
    model = models.mobilenet_v2()
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model_path = os.path.join(config.MODEL_SAVE_DIR, 'mobilenet_best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 4. 加载数据
    with open(config.TEST, 'r') as f:
        test_samples = json.load(f)

    print(f">>> 开始评估指标，总样本数: {len(test_samples)}")

    # 用于计算指标的列表
    all_labels = []
    all_preds = []
    all_scores = []
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for item in test_samples:
            img_path = os.path.join(config.TEST_IMG_DIR, item['file_name'])
            img_cv = cv2.imread(img_path)
            if img_cv is None: continue

            # 裁剪与转换
            x, y, w, h = map(int, item['bbox'])
            h_img, w_img = img_cv.shape[:2]
            crop = img_cv[max(0, y):min(h_img, y + h), max(0, x):min(w_img, x + w)]
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            input_tensor = test_transform(img_pil).unsqueeze(0).to(device)

            # 模型推理
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # --- 计算 Top-1 和 Top-5 ---
            true_label_idx = int(id2label[item['tiger_id']])
            maxk = min(5, num_classes)
            _, topk_preds = outputs.topk(maxk, 1, True, True)
            topk_preds = topk_preds.t()

            total += 1
            all_labels.append(true_label_idx)
            all_preds.append(topk_preds[0].item())  # Top-1 预测
            all_scores.append(probs.cpu().numpy()[0])  # 记录所有类别的概率值用于 mAP

            # 统计正确数
            if true_label_idx == topk_preds[0]:
                top1_correct += 1
            if true_label_idx in topk_preds:
                top5_correct += 1

    # --- 5. 计算最终指标 ---
    top1_acc = top1_correct / total
    top5_acc = top5_correct / total

    # 计算 mAP (Mean Average Precision)
    # 将标签二值化以计算每一类的 AP
    Y_true_bin = label_binarize(all_labels, classes=list(range(num_classes)))
    all_scores = np.array(all_scores)
    # 使用 macro 平均计算 mAP
    mAP = average_precision_score(Y_true_bin, all_scores, average="macro")

    print("\n" + "=" * 45)
    print(f"深度学习模型高级评估报告")
    print(f"Top-1 Accuracy: {top1_acc:.2%}")
    print(f"Top-5 Accuracy: {top5_acc:.2%}")
    print(f"mAP (Mean Average Precision): {mAP:.4f}")
    print(f"Total Samples: {total}")
    print("=" * 45 + "\n")

    # --- 6. 绘制混淆矩阵 ---
    draw_confusion_matrix(all_labels, all_preds, num_classes)


def draw_confusion_matrix(y_true, y_pred, num_classes):
    print(">>> 正在生成混淆矩阵图片...")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # 如果 ID 太多，不显示具体的数字标注 (annot=False)
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')

    plt.title('Confusion Matrix of Tiger Re-ID')
    plt.xlabel('Predicted Label Index')
    plt.ylabel('True Label Index')

    output_path = 'MobileNetV2confusion_matrix.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ 混淆矩阵已保存至: {output_path}")


if __name__ == '__main__':
    evaluate_deep()