import json
import os
import cv2
import matplotlib

# 强制使用 Agg 后端，避免 PyCharm/远程环境的显示报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import config


def load_data(mapping_json):
    """加载并解析数据"""
    if not os.path.exists(mapping_json):
        print(f"❌ 找不到文件: {mapping_json}")
        return None

    with open(mapping_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tiger_ids = []
    aspect_ratios = []

    print(f">>> 正在读取 {len(data)} 张图片信息...")
    for item in data:
        # ID 统计
        tiger_ids.append(item['tiger_id'])

        # 宽高比统计 (Width / Height)
        img_path = os.path.join(config.TEST_IMG_DIR, item['file_name'])
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            aspect_ratios.append(w / h)

    return tiger_ids, aspect_ratios


def plot_id_distribution(tiger_ids, save_path="id_distribution.png"):
    """绘制训练集 ID 分布图"""
    id_counts = Counter(tiger_ids)
    # 按 ID 数字顺序排序
    sorted_items = sorted(id_counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
    ids, counts = zip(*sorted_items)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=list(ids), y=list(counts), color="royalblue")

    # --- 核心修改：稀疏显示横坐标 ---
    # 每隔 5 个 ID 显示一个标签，防止拥挤
    n = 5
    for i, label in enumerate(ax.get_xticklabels()):
        if i % n != 0:
            label.set_visible(False)

    plt.title("Tiger ID Distribution (Samples per ID)", fontsize=14)
    plt.xlabel("Tiger ID (Showing every 5th ID)", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ ID 分布图已保存至: {save_path}")


def plot_aspect_ratio(aspect_ratios, save_path="aspect_ratio_distribution.png"):
    """绘制宽高比分布图"""
    plt.figure(figsize=(10, 6))
    sns.histplot(aspect_ratios, bins=40, kde=True, color="seagreen", edgecolor='white')

    # 添加 1:1 标准线
    plt.axvline(1.0, color='red', linestyle='--', label='Square (1:1)')

    # 计算平均值
    avg_ratio = np.mean(aspect_ratios)
    plt.axvline(avg_ratio, color='orange', linestyle='-', label=f'Mean: {avg_ratio:.2f}')

    plt.title("Image Aspect Ratio Distribution (Width / Height)", fontsize=14)
    plt.xlabel("Aspect Ratio", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(axis='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 宽高比分布图已保存至: {save_path}")


cfg = config.TEST

if __name__ == "__main__":
    # 使用你生成的训练集映射文件
    ids_data, ratios_data = load_data(cfg)

    if ids_data:
        # plot_id_distribution(ids_data)
        # plot_aspect_ratio(ratios_data)

        print("\n" + "=" * 30)
        print(f"数据概览：")
        print(f"总图片数: {len(ids_data)}")
        print(f"老虎总数: {len(set(ids_data))}")
        print(f"平均宽高比: {np.mean(ratios_data):.2f}")
        print("=" * 30)
