import cv2
import numpy as np
import joblib
import os
from sklearn.mixture import GaussianMixture

class FisherVectorExtractor:
    def __init__(self, n_kernels=64):
        self.n_kernels = n_kernels
        self.sift = cv2.SIFT_create()
        self.gmm = None

    def get_descriptors_from_image(self, img):
        """输入图片数组(numpy)，返回SIFT描述子"""
        if img is None: return None
        # 如果是彩色图，转灰度
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        kps, des = self.sift.detectAndCompute(gray, None)
        return des

    # --- 关键修改：将 train_gmm 改名为 fit 以匹配 main.py ---
    def fit(self, image_list):
        """
        训练GMM。
        image_list: 包含图片数组的列表 (即已经裁剪好的老虎图片)
        """
        print(f"正在提取特征用于 GMM 训练 (共 {len(image_list)} 张图片)...")
        all_descriptors = []

        # 限制用于训练GMM的特征总数，防止内存爆炸
        # 当 K=128 时，建议增加采样图片数量以获得更好的聚类效果
        sample_images = image_list[:1000] if len(image_list) > 1000 else image_list

        for img in sample_images:
            des = self.get_descriptors_from_image(img)
            if des is not None:
                all_descriptors.append(des)

        if not all_descriptors:
            raise ValueError("没有提取到任何特征，请检查图片是否正确加载！")

        all_descriptors = np.vstack(all_descriptors)
        print(f"提取完成，特征总数: {all_descriptors.shape}。开始训练 GMM (K={self.n_kernels})...")

        # 使用更稳健的协方差类型和初始化
        self.gmm = GaussianMixture(n_components=self.n_kernels, covariance_type='diag', random_state=42)
        self.gmm.fit(all_descriptors)
        print("GMM 训练完成！")

    def fisher_vector(self, img):
        """生成单张图片的 FV 向量"""
        descriptors = self.get_descriptors_from_image(img)

        # 如果这张图提不出特征（比如太模糊），返回零向量
        if descriptors is None:
            return np.zeros(2 * 128 * self.n_kernels, dtype=np.float32)

        # GMM 预测
        gmm_likelihood = self.gmm.predict_proba(descriptors)

        means = self.gmm.means_
        covars = self.gmm.covariances_
        weights = self.gmm.weights_

        T = descriptors.shape[0]
        D = descriptors.shape[1]
        K = self.n_kernels

        fv_means = np.zeros((K, D))
        fv_covars = np.zeros((K, D))

        # 向量化计算加速
        for k in range(K):
            # 获取属于第k个聚类中心的权重
            weight_k = gmm_likelihood[:, k].reshape(T, 1)
            s0 = np.sum(weight_k)
            s1 = np.dot(weight_k.T, descriptors)
            s2 = np.dot(weight_k.T, descriptors ** 2)

            # 防止数值崩溃（由于 K=128 时聚类可能更细，s0 可能很小）
            if s0 < 1e-6:
                continue

            fv_means[k] = (s1 - s0 * means[k]) / (np.sqrt(weights[k]) * np.sqrt(covars[k]) * T)
            fv_covars[k] = (s2 - 2 * s1 * means[k] + s0 * (means[k] ** 2 - covars[k])) / (
                        np.sqrt(2 * weights[k]) * covars[k] * T)

        fv = np.concatenate([fv_means.flatten(), fv_covars.flatten()])

        # 归一化：Power norm + L2 norm
        fv = np.sign(fv) * np.sqrt(np.abs(fv))
        norm = np.linalg.norm(fv)
        if norm > 1e-8:
            fv = fv / norm

        return fv.astype(np.float32)

    def save_gmm(self, path):
        joblib.dump(self.gmm, path)

    def load_gmm(self, path):
        self.gmm = joblib.load(path)