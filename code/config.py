import os

# ================= 路径配置 (请保持你的根目录不变) =================
DATA_ROOT = r'/tmp/pycharm_project_848/datasets/ATRW'

# 图片目录 (通常 keypoint_train.json 里的图片就在 images/train 下)
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'images', 'train')
VAL_IMG_DIR = os.path.join(DATA_ROOT, 'images', 'val')
TEST_IMG_DIR = os.path.join(DATA_ROOT, 'images', 'test')

# ================= 标注文件路径 (这里修改为你的新文件名) =================
# 使用包含关键点和BBox信息的文件
TRAIN_JSON = os.path.join(DATA_ROOT, 'annotations', 'keypoint_train.json')
VAL_JSON = os.path.join(DATA_ROOT, 'annotations', 'keypoint_val.json')
TRAINVAL_JSON = os.path.join(DATA_ROOT, 'annotations', 'keypoint_trainval.json')
TEST_JSON = os.path.join(DATA_ROOT, 'annotations', 'pose_tiger02_test.json')

# 处理过后的mapping文件
TRAIN = os.path.join(DATA_ROOT, 'annotations', 'mapped_train.json')
VAL = os.path.join(DATA_ROOT, 'annotations', 'mapped_val.json')
TRAINVAL = os.path.join(DATA_ROOT, 'annotations', 'mapped_trainval.json')
TEST = os.path.join(DATA_ROOT, 'annotations', 'mapped_test.json')

REID_SPLITS_JSON = os.path.join(DATA_ROOT, 'annotations', 'reid-splits.json')
TRAIN_CSV = os.path.join(DATA_ROOT, 'annotations', 'reid_list_train.csv')

# 模型保存路径
MODEL_SAVE_DIR = './checkpoints'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ================= 训练参数 =================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
GMM_KERNELS = 64
SIFT_DIM = 128