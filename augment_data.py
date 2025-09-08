"""
3D点云数据增强脚本
用于人脸识别任务的数据扩充
"""

import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import json
import random
from sklearn.model_selection import train_test_split

class PointCloudAugmenter:
    """点云数据增强器"""
    
    def __init__(self, seed=42):
        """
        初始化增强器
        Args:
            seed: 随机种子，确保可重复性
        """
        np.random.seed(seed)
        random.seed(seed)
        
    def random_rotation(self, points, angle_range=10):
        """
        随机旋转点云
        Args:
            points: (N, 3) 点云
            angle_range: 旋转角度范围（度）
        """
        angles = np.random.uniform(-angle_range, angle_range, 3) * np.pi / 180
        
        # 旋转矩阵
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(angles[0]), -np.sin(angles[0])],
                      [0, np.sin(angles[0]), np.cos(angles[0])]])
        
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                      [0, 1, 0],
                      [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                      [np.sin(angles[2]), np.cos(angles[2]), 0],
                      [0, 0, 1]])
        
        R = Rz @ Ry @ Rx
        return points @ R.T
    
    def random_scale(self, points, scale_range=0.1):
        """
        随机缩放
        Args:
            points: (N, 3) 点云
            scale_range: 缩放范围，如0.1表示0.9-1.1倍
        """
        scale = np.random.uniform(1 - scale_range, 1 + scale_range)
        return points * scale
    
    def random_noise(self, points, noise_level=0.01):
        """
        添加高斯噪声
        Args:
            points: (N, 3) 点云
            noise_level: 噪声标准差
        """
        noise = np.random.normal(0, noise_level, points.shape)
        return points + noise
    
    def random_dropout(self, points, dropout_rate=0.05):
        """
        随机丢弃点
        Args:
            points: (N, 3) 点云
            dropout_rate: 丢弃比例
        """
        n_points = len(points)
        n_keep = int(n_points * (1 - dropout_rate))
        
        # 随机选择保留的点
        keep_indices = np.random.choice(n_points, n_keep, replace=False)
        dropped_points = points[keep_indices]
        
        # 如果点数减少了，通过重采样补齐
        if len(dropped_points) < n_points:
            resample_indices = np.random.choice(len(dropped_points), 
                                              n_points - len(dropped_points), 
                                              replace=True)
            dropped_points = np.vstack([dropped_points, dropped_points[resample_indices]])
        
        return dropped_points
    
    def random_translation(self, points, translation_range=0.05):
        """
        随机平移
        Args:
            points: (N, 3) 点云
            translation_range: 平移范围
        """
        translation = np.random.uniform(-translation_range, translation_range, 3)
        return points + translation
    
    def mirror_flip(self, points, axis=0):
        """
        镜像翻转（对人脸特别有用）
        Args:
            points: (N, 3) 点云
            axis: 翻转轴 (0=X轴左右翻转, 1=Y轴, 2=Z轴)
        """
        flipped = points.copy()
        flipped[:, axis] = -flipped[:, axis]
        return flipped
    
    def elastic_deformation(self, points, alpha=0.1, sigma=0.05):
        """
        弹性变形（模拟表情变化）
        Args:
            points: (N, 3) 点云
            alpha: 变形强度
            sigma: 平滑参数
        """
        # 生成随机位移场
        displacement = np.random.randn(*points.shape) * alpha
        
        # 简单的高斯平滑（可选）
        from scipy.ndimage import gaussian_filter1d
        for i in range(3):
            displacement[:, i] = gaussian_filter1d(displacement[:, i], sigma * len(points))
        
        return points + displacement
    
    def augment_single(self, points, config):
        """
        对单个点云应用一系列增强
        Args:
            points: (N, 3) 点云
            config: 增强配置字典
        Returns:
            增强后的点云
        """
        augmented = points.copy()
        
        # 按配置应用各种增强
        if config.get('rotation', False):
            augmented = self.random_rotation(augmented, config.get('rotation_range', 10))
        
        if config.get('scale', False):
            augmented = self.random_scale(augmented, config.get('scale_range', 0.1))
        
        if config.get('noise', False):
            augmented = self.random_noise(augmented, config.get('noise_level', 0.01))
        
        if config.get('dropout', False):
            augmented = self.random_dropout(augmented, config.get('dropout_rate', 0.05))
        
        if config.get('translation', False):
            augmented = self.random_translation(augmented, config.get('translation_range', 0.05))
        
        if config.get('mirror', False) and np.random.rand() > 0.5:
            augmented = self.mirror_flip(augmented)
        
        if config.get('elastic', False) and np.random.rand() > 0.7:
            augmented = self.elastic_deformation(augmented, 
                                                config.get('elastic_alpha', 0.1),
                                                config.get('elastic_sigma', 0.05))
        
        return augmented.astype(np.float32)
    
    def augment_batch(self, points, n_augmentations, config):
        """
        生成多个增强版本
        Args:
            points: 原始点云
            n_augmentations: 生成的增强数量
            config: 增强配置
        Returns:
            增强后的点云列表
        """
        augmented_list = []
        for _ in range(n_augmentations):
            augmented = self.augment_single(points, config)
            augmented_list.append(augmented)
        return augmented_list


class DataOrganizer:
    """数据组织和划分工具"""
    
    def __init__(self, source_dir, output_dir, test_size=0.15, val_size=0.15):
        """
        Args:
            source_dir: 原始NPZ文件目录
            output_dir: 输出目录
            test_size: 测试集比例
            val_size: 验证集比例
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.val_size = val_size
        
    def organize_by_identity(self):
        """
        按身份组织文件（如果文件名包含身份信息）
        假设文件名格式：person_001_xxx.npz
        """
        identity_map = {}
        
        for npz_file in self.source_dir.glob('*.npz'):
            # 提取身份ID（根据您的文件命名调整）
            # 示例：person_001_scan1.npz -> person_001
            parts = npz_file.stem.split('_')
            if len(parts) >= 2:
                identity = '_'.join(parts[:2])  # person_001
            else:
                identity = npz_file.stem
            
            if identity not in identity_map:
                identity_map[identity] = []
            identity_map[identity].append(npz_file)
        
        return identity_map
    
    def split_data(self, file_list, labels=None):
        """
        划分训练/验证/测试集
        Args:
            file_list: 文件列表
            labels: 对应的标签（可选）
        """
        # 如果没有标签，简单随机划分
        if labels is None:
            # 首先分出测试集
            n_total = len(file_list)
            n_test = int(n_total * self.test_size)
            n_val = int(n_total * self.val_size)
            n_train = n_total - n_test - n_val
            
            # 随机打乱
            indices = np.arange(n_total)
            np.random.shuffle(indices)
            
            # 划分
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
            
            train_files = [file_list[i] for i in train_indices]
            val_files = [file_list[i] for i in val_indices]
            test_files = [file_list[i] for i in test_indices]
            
            return {
                'train': (train_files, None),
                'val': (val_files, None),
                'test': (test_files, None)
            }
        
        else:
            # 有标签时使用分层划分
            train_val_files, test_files, train_val_labels, test_labels = train_test_split(
                file_list, labels,
                test_size=self.test_size,
                random_state=42,
                stratify=labels
            )
            
            # 从剩余的划分验证集
            val_ratio = self.val_size / (1 - self.test_size)
            train_files, val_files, train_labels, val_labels = train_test_split(
                train_val_files, train_val_labels,
                test_size=val_ratio,
                random_state=42,
                stratify=train_val_labels
            )
            
            return {
                'train': (train_files, train_labels),
                'val': (val_files, val_labels),
                'test': (test_files, test_labels)
            }
    
    def setup_directories(self):
        """创建目录结构"""
        dirs = [
            self.output_dir / 'train',
            self.output_dir / 'train_augmented',
            self.output_dir / 'val',
            self.output_dir / 'test'
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return dirs


def process_and_augment(source_dir, output_dir, augment_config=None, n_augmentations=5):
    """
    主处理函数：组织数据并进行增强
    
    Args:
        source_dir: 原始NPZ文件目录
        output_dir: 输出目录
        augment_config: 增强配置
        n_augmentations: 每个训练样本生成的增强数量
    """
    
    # 默认增强配置
    if augment_config is None:
        augment_config = {
            'rotation': True,
            'rotation_range': 10,  # ±10度
            'scale': True,
            'scale_range': 0.1,     # 0.9-1.1倍
            'noise': True,
            'noise_level': 0.005,   # 适中的噪声
            'dropout': True,
            'dropout_rate': 0.05,   # 丢弃5%的点
            'translation': True,
            'translation_range': 0.03,
            'mirror': True,         # 人脸镜像翻转
            'elastic': False        # 弹性变形（可选）
        }
    
    print("=" * 50)
    print("3D点云数据组织与增强")
    print("=" * 50)
    
    # 确保输出目录是Path对象
    output_dir = Path(output_dir)
    
    # 初始化
    organizer = DataOrganizer(source_dir, output_dir)
    augmenter = PointCloudAugmenter()
    
    # 创建目录结构
    organizer.setup_directories()
    
    # 获取所有NPZ文件
    all_files = list(Path(source_dir).glob('*.npz'))
    print(f"找到 {len(all_files)} 个NPZ文件")
    
    # 检查是否每个文件代表不同的人
    # 如果您的1030个文件是1030个不同的人，我们就不使用身份标签
    identity_map = organizer.organize_by_identity()
    
    # 判断是否是多样本per身份的情况
    has_multiple_samples = any(len(files) > 1 for files in identity_map.values())
    
    if has_multiple_samples:
        print(f"检测到 {len(identity_map)} 个不同身份（某些身份有多个样本）")
        # 创建标签
        labels = []
        file_list = []
        for identity_id, (identity, files) in enumerate(identity_map.items()):
            for f in files:
                file_list.append(f)
                labels.append(identity_id)
        
        # 使用分层划分
        splits = organizer.split_data(file_list, labels)
    else:
        # 每个文件是独立的样本，不使用分层
        print(f"每个文件代表独立的样本（共{len(all_files)}个）")
        file_list = all_files
        
        # 简单随机划分，不使用标签
        splits = organizer.split_data(file_list, labels=None)
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(splits['train'][0])} 个文件")
    print(f"  验证集: {len(splits['val'][0])} 个文件")
    print(f"  测试集: {len(splits['test'][0])} 个文件")
    
    # 处理统计
    stats = {
        'original_count': len(all_files),
        'train_count': len(splits['train'][0]),
        'val_count': len(splits['val'][0]),
        'test_count': len(splits['test'][0]),
        'augmented_count': 0,
        'augmentation_config': augment_config,
        'has_identity_labels': has_multiple_samples
    }
    
    # 处理各个数据集
    for split_name, (files, split_labels) in splits.items():
        print(f"\n处理 {split_name} 集...")
        
        for i, file_path in enumerate(tqdm(files, desc=f"Processing {split_name}")):
            # 加载原始数据
            data = np.load(file_path)
            points = data['points']
            normals = data.get('normals', None)
            
            # 获取文件名
            file_stem = Path(file_path).stem
            
            if split_name == 'train':
                # 保存原始训练数据
                output_path = output_dir / 'train' / f"{file_stem}.npz"
                save_data = {'points': points}
                if normals is not None:
                    save_data['normals'] = normals
                if split_labels is not None:
                    save_data['label'] = split_labels[i]
                np.savez_compressed(output_path, **save_data)
                
                # 生成增强数据
                augmented_points_list = augmenter.augment_batch(
                    points, n_augmentations, augment_config
                )
                
                for aug_idx, aug_points in enumerate(augmented_points_list):
                    aug_output_path = output_dir / 'train_augmented' / f"{file_stem}_aug{aug_idx}.npz"
                    save_data = {'points': aug_points}
                    if normals is not None:
                        # 对法向量应用相同的旋转（如果有）
                        save_data['normals'] = normals  # 简化处理，实际应用时需要同步变换
                    if split_labels is not None:
                        save_data['label'] = split_labels[i]
                    np.savez_compressed(aug_output_path, **save_data)
                    stats['augmented_count'] += 1
            
            else:
                # 验证集和测试集不增强，直接复制
                output_path = output_dir / split_name / f"{file_stem}.npz"
                save_data = {'points': points}
                if normals is not None:
                    save_data['normals'] = normals
                if split_labels is not None:
                    save_data['label'] = split_labels[i]
                np.savez_compressed(output_path, **save_data)
    
    # 保存统计信息
    stats_path = output_dir / 'data_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 创建文件列表（方便后续加载）
    for split_name in ['train', 'train_augmented', 'val', 'test']:
        split_dir = output_dir / split_name
        file_list = [str(f.relative_to(output_dir)) for f in split_dir.glob('*.npz')]
        list_path = output_dir / f'{split_name}_files.txt'
        with open(list_path, 'w') as f:
            f.write('\n'.join(file_list))
    
    print("\n" + "=" * 50)
    print("处理完成！")
    print(f"输出目录: {output_dir}")
    print(f"统计信息:")
    print(f"  原始训练样本: {len(splits['train'][0])}")
    print(f"  增强后训练样本: {len(splits['train'][0]) * (1 + n_augmentations)}")
    print(f"  验证样本: {len(splits['val'][0])}")
    print(f"  测试样本: {len(splits['test'][0])}")
    print("=" * 50)
    
    return stats


def visualize_augmentation_effects(npz_file, output_dir='augmentation_preview'):
    """
    可视化增强效果（用于调试和验证）
    """
    import matplotlib.pyplot as plt
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # 加载原始数据
    data = np.load(npz_file)
    original = data['points']
    
    # 创建增强器
    augmenter = PointCloudAugmenter()
    
    # 不同的增强方法
    augmentations = {
        'Original': original,
        'Rotated': augmenter.random_rotation(original),
        'Scaled': augmenter.random_scale(original),
        'Noisy': augmenter.random_noise(original),
        'Dropout': augmenter.random_dropout(original),
        'Mirrored': augmenter.mirror_flip(original),
    }
    
    # 创建可视化
    fig = plt.figure(figsize=(15, 10))
    
    for idx, (name, points) in enumerate(augmentations.items()):
        ax = fig.add_subplot(2, 3, idx + 1)
        ax.scatter(points[:, 0], points[:, 1], s=0.5, alpha=0.5)
        ax.set_title(name)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/augmentation_preview.png", dpi=150)
    plt.close()
    
    print(f"增强效果预览已保存到: {output_dir}/augmentation_preview.png")


if __name__ == "__main__":
    # ========== 使用示例 ==========
    
    # 1. 基本使用
    process_and_augment(
        source_dir='/home/jz97/3d_face_repo/obj_data_1030_pointcloud',  # 您的NPZ文件目录
        output_dir='/home/jz97/3d_face_repo/augmented_data',         # 输出目录
        n_augmentations=5                    # 每个样本生成5个增强版本
    )
    
    # 2. 自定义增强配置
    custom_config = {
        'rotation': True,
        'rotation_range': 15,    # 更大的旋转
        'scale': True,
        'scale_range': 0.15,     # 更大的缩放范围
        'noise': True,
        'noise_level': 0.01,
        'dropout': False,        # 不使用dropout
        'mirror': True,
        'elastic': False
    }
    
    # process_and_augment(
    #     source_dir='processed_pointclouds',
    #     output_dir='augmented_data_custom',
    #     augment_config=custom_config,
    #     n_augmentations=3
    # )
    
    # 3. 预览增强效果（可选）
    # visualize_augmentation_effects(
    #     'processed_pointclouds/sample.npz',
    #     'augmentation_preview'
    # )