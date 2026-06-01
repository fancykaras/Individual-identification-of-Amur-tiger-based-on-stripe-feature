import torch
import os
import json
import cv2
import config
import numpy as np
from fv_utils import FisherVectorExtractor
from dataset import ATRWDataset
from main import TigerMLP

# 用于指标计算和绘图
import matplotlib

matplotlib.use('Agg')  # 解决服务器环境/PyCharm 远程调试的显示报错
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载训练时的 ID 映射表
    mapping_path = os.path.join(config.MODEL_SAVE_DIR, 'id_mapping.json')
    if not os.path.exists(mapping_path):
        print(f"❌ 错误: 找不到映射文件 {mapping_path}")
        return

    with open(mapping_path, 'r') as f:
        global_id2label = json.load(f)

    index_to_tiger_id = {int(v): k for k, v in global_id2label.items()}
    num_classes = len(global_id2label)

    # 2. 加载特征提取器和模型
    extractor = FisherVectorExtractor(n_kernels=config.GMM_KERNELS)
    extractor.load_gmm(os.path.join(config.MODEL_SAVE_DIR, 'gmm.pkl'))

    input_dim = 2 * config.SIFT_DIM * config.GMM_KERNELS
    model = TigerMLP(input_dim, num_classes).to(device)

    model_weights = os.path.join(config.MODEL_SAVE_DIR, 'mlp_final.pth')
    if os.path.exists(model_weights):
        model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))
    model.eval()

    # 3. 加载测试集并过滤
    print(">>> 正在加载测试集并提取 FV 特征...")
    test_dataset = ATRWDataset(mode='test', extractor=extractor, config=config, mapping_json=config.TEST)

    # 过滤掉训练集没见过的老虎 ID
    test_dataset.samples = [s for s in test_dataset.samples if s['tiger_id'] in global_id2label]
    test_dataset.id2label = global_id2label

    # --- 初始化指标统计变量 ---
    all_true_labels = []
    all_pred_labels = []
    all_scores = []
    top1_correct = 0
    top5_correct = 0
    total = 0
    visualization_results = []

    print(f">>> 开始计算指标，过滤后样本数: {len(test_dataset)}")

    with torch.no_grad():
        for i in range(len(test_dataset)):
            # labels 这里是 int，不要用 .item()
            inputs, labels = test_dataset[i]
            inputs = inputs.unsqueeze(0).to(device)
            outputs = model(inputs)

            # 概率转换
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # Top-1 & Top-5 逻辑
            maxk = min(5, num_classes)
            _, topk_preds = outputs.topk(maxk, 1, True, True)
            topk_list = topk_preds.cpu().numpy().flatten().tolist()

            true_idx = labels  # 修复 AttributeError
            pred_idx = topk_list[0]

            total += 1
            all_true_labels.append(true_idx)
            all_pred_labels.append(pred_idx)
            all_scores.append(probs.cpu().numpy()[0])

            # 统计 Top-K
            if true_idx == pred_idx:
                top1_correct += 1
            if true_idx in topk_list:
                top5_correct += 1

            # 收集用于可视化对比的样本 (前10个)
            if len(visualization_results) < 10:
                visualization_results.append({
                    'sample': test_dataset.samples[i],
                    'pred_id': index_to_tiger_id[pred_idx],
                    'conf': probs[0, pred_idx].item(),
                    'is_correct': (true_idx == pred_idx)
                })

    # --- 4. 计算并输出最终指标 ---
    accuracy = top1_correct / total if total > 0 else 0
    top5_acc = top5_correct / total if total > 0 else 0

    # 计算 mAP
    y_true_bin = label_binarize(all_true_labels, classes=list(range(num_classes)))
    mAP = average_precision_score(y_true_bin, np.array(all_scores), average="macro")

    print("\n" + "=" * 45)
    print(f"SIFT+FV+MLP ")
    print(f"Top-1 Accuracy : {accuracy:.2%}")
    print(f"Top-5 Accuracy: {top5_acc:.2%}")
    print(f"mAP (Mean Average Precision): {mAP:.4f}")
    print(f"Deviation Rate: {1 - accuracy:.2%}")
    print("=" * 45 + "\n")

    # --- 5. 执行绘图逻辑 ---
    plot_confusion_matrix(all_true_labels, all_pred_labels, num_classes)
    draw_results(visualization_results)
    cmc_data = plot_cmc_curve(all_true_labels, all_scores, max_rank=20)
    mAP_value = plot_pr_curve(all_true_labels, all_scores, num_classes, save_path='pr_resnet50.png')

def plot_confusion_matrix(y_true, y_pred, num_classes):
    """绘制混淆矩阵"""
    print(">>> 正在生成混淆矩阵...")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    plt.figure(figsize=(15, 12))
    # 老虎 ID 通常较多，建议不显示数值注释 (annot=False)
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)

    plt.title('Confusion Matrix - SIFT+FV+MLP', fontsize=16)
    plt.xlabel('Predicted Label Index', fontsize=12)
    plt.ylabel('True Label Index', fontsize=12)

    output_name = 'confusion_matrix_traditional.png'
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存至: {output_name}")


def plot_cmc_curve(all_true_labels, all_scores, max_rank=20, save_path='cmc_curve.png'):
    """
    绘制 CMC 曲线
    all_true_labels: 真实标签列表 (N,)
    all_scores: 模型输出的概率矩阵 (N, num_classes)
    max_rank: 计算到 Rank-几
    """
    num_samples = len(all_true_labels)
    # 转换为 numpy 方便处理
    all_scores = np.array(all_scores)
    all_true_labels = np.array(all_true_labels)

    # 存储每个 Rank 的准确率
    match_counts = np.zeros(max_rank)

    for i in range(num_samples):
        true_label = all_true_labels[i]
        # 获取第 i 个样本的预测分值排序 (从大到小)
        # argsort 返回的是索引，[::-1] 进行翻转
        sorted_indices = np.argsort(all_scores[i])[::-1]

        # 找到真实标签在排序中的位置 (rank)
        # np.where 返回的是元组，取 [0][0] 得到具体的索引位置
        rank = np.where(sorted_indices == true_label)[0][0]

        # 如果 rank 小于 max_rank，说明它在 CMC 统计范围内
        if rank < max_rank:
            # 该样本对 Rank_(rank+1) 及其以后的所有 rank 都有贡献
            match_counts[rank:] += 1

    # 计算百分比
    cmc_scores = match_counts / num_samples

    # 绘图
    plt.figure(figsize=(10, 6))
    ranks = np.arange(1, max_rank + 1)
    plt.plot(ranks, cmc_scores, marker='o', linestyle='-', color='b', linewidth=2, markersize=5)

    # 设置图表格式
    plt.title('Cumulative Matching Characteristics (CMC) Curve', fontsize=14)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Identification Rate', fontsize=12)
    plt.xticks(ranks)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim([0, 1.05])

    # 在图中标准 Rank-1, Rank-5, Rank-10 的数值
    for r in [1, 5, 10]:
        if r <= max_rank:
            plt.text(r, cmc_scores[r - 1], f'R{r}:{cmc_scores[r - 1]:.2%}',
                     va='bottom', ha='right', fontsize=10)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ CMC 曲线已保存至: {save_path}")
    return cmc_scores


def plot_pr_curve(all_true_labels, all_scores, num_classes, save_path='pr_curve.png'):
    """
    绘制多分类任务的 Macro-average P-R 曲线
    all_true_labels: 真实标签列表 (N,)
    all_scores: 模型输出的概率矩阵 (N, num_classes)
    num_classes: 总类别数
    """
    # 1. 将标签进行二值化 (One-hot 编码)
    y_true_bin = label_binarize(all_true_labels, classes=list(range(num_classes)))
    all_scores = np.array(all_scores)

    # 2. 计算每一类的 Precision 和 Recall
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], all_scores[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], all_scores[:, i])

    # 3. 计算 Macro-average (宏平均)
    # 首先合并所有的 recall 轴
    all_recall = np.unique(np.concatenate([recall[i] for i in range(num_classes)]))

    # 然后在该轴上对 precision 进行插值
    mean_precision = np.zeros_like(all_recall)
    for i in range(num_classes):
        mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])

    mean_precision /= num_classes
    macro_auc = average_precision_score(y_true_bin, all_scores, average="macro")

    # 4. 绘图
    plt.figure(figsize=(10, 7))
    plt.plot(all_recall, mean_precision, color='darkorange', lw=2,
             label=f'Macro-average P-R curve (area = {macro_auc:.3f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve (Multi-class)', fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ P-R 曲线已保存至: {save_path}")
    return macro_auc

def draw_results(results):
    """绘制单张图片的识别结果对比"""
    print(">>> 正在生成样本可视化分析图...")
    for i, res in enumerate(results):
        sample = res['sample']
        img_path = os.path.join(config.TEST_IMG_DIR, sample['file_name'])
        img = cv2.imread(img_path)
        if img is None: continue

        x, y, w, h = map(int, sample['bbox'])
        color = (0, 255, 0) if res['is_correct'] else (0, 0, 255)

        # 绘制识别框和文字
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        pred_text = f"Pred: {res['pred_id']} ({res['conf']:.1%})"
        true_text = f"True: {sample['tiger_id']}"

        # 绘制背景条使文字更清晰
        cv2.rectangle(img, (x, y - 65), (x + 300, y), color, -1)
        cv2.putText(img, pred_text, (x + 5, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, true_text, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imwrite(f"test_analysis_traditional_{i}.jpg", img)
    print(f"✅ 可视化分析图已生成。")


if __name__ == '__main__':
    evaluate()