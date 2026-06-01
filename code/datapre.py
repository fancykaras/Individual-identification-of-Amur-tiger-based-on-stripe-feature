import json
import csv
import os
import config

def process_single_json(csv_mapping, json_path, output_path):
    """
    处理单个 JSON 文件，将其与 CSV 比对并生成新的映射文件
    """
    if not os.path.exists(json_path):
        print(f"⚠️ 跳过：找不到文件 {json_path}")
        return

    print(f"正在处理: {json_path} -> {output_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 建立当前 JSON 的 filename -> image_id 映射
    name_to_id = {}
    for img in data.get('images', []):
        fname = img.get('filename') or img.get('file_name')
        if fname:
            name_to_id[fname] = img['id']

    # 2. 建立当前 JSON 的 image_id -> bbox 映射
    id_to_bbox = {}
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in id_to_bbox:
            id_to_bbox[img_id] = ann['bbox']

    # 3. 比对并构建结果
    mapped_results = {}  # 使用字典确保 image_id 唯一

    for fname, tiger_id in csv_mapping.items():
        # 检查该图片是否出现在当前的这个 JSON 中
        if fname in name_to_id:
            img_id = name_to_id[fname]
            bbox = id_to_bbox.get(img_id, [0, 0, 0, 0])  # 若无 bbox 则默认全 0

            # 填充信息
            mapped_results[img_id] = {
                "image_id": img_id,
                "tiger_id": tiger_id,
                "file_name": fname,
                "bbox": bbox
            }

    # 4. 写入新的 JSON
    final_list = list(mapped_results.values())
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, indent=4, ensure_ascii=False)

    print(f"✅ 完成！生成条目: {len(final_list)}")


def run_conversion():
    # 配置路径
    csv_path = config.TRAIN_CSV

    # 输入与输出的对应关系
    task_pairs = [
        (config.TRAIN_JSON, os.path.join(config.DATA_ROOT, 'annotations', 'mapped_train.json')),
        (config.VAL_JSON, os.path.join(config.DATA_ROOT, 'annotations', 'mapped_val.json')),
        (config.TRAINVAL_JSON, os.path.join(config.DATA_ROOT, 'annotations', 'mapped_trainval.json')),
        (config.TEST_JSON, os.path.join(config.DATA_ROOT, 'annotations', 'mapped_test.json'))
    ]

    # 1. 首先加载 CSV (全量的老虎 ID 查找表)
    csv_mapping = {}
    print(f"正在读取 CSV 查找表: {csv_path}...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2: continue
            # 格式：tiger_id, file_name
            t_id, f_name = row[0].strip(), row[1].strip()
            csv_mapping[f_name] = t_id

    # 2. 依次处理每个 JSON
    print("-" * 30)
    for input_j, output_j in task_pairs:
        process_single_json(csv_mapping, input_j, output_j)
    print("-" * 30)
    print("所有任务处理完毕！")


if __name__ == "__main__":
    run_conversion()