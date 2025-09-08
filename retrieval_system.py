"""
3D人脸检索系统
使用训练好的特征提取器构建检索系统并评估性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import pickle
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

# =====================================
# 第1部分：加载训练好的模型
# =====================================

# 从训练代码导入模型定义
from train_feature_extractor import FaceFeatureExtractor, PointNet2Feature

class FeatureExtractor:
    """封装特征提取功能"""
    
    def __init__(self, model_path='face_feature_extractor_final.pth', device='cuda'):
        """
        Args:
            model_path: 训练好的模型路径
            device: 运行设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = FaceFeatureExtractor(feature_dim=512, freeze_layers=0)
        
        # 加载权重（处理PyTorch 2.6的兼容性问题）
        try:
            # 首先尝试安全加载
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except:
            # 如果失败，使用旧版本方式（信任源文件）
            print("  使用兼容模式加载...")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ 模型加载成功")
        print(f"  设备: {self.device}")
    
    def extract_features(self, point_cloud):
        """
        提取单个或批量点云的特征
        
        Args:
            point_cloud: numpy数组 (N, 3) 或 (B, N, 3)
        
        Returns:
            features: (512,) 或 (B, 512) 的特征向量
        """
        # 确保输入格式正确
        if isinstance(point_cloud, np.ndarray):
            point_cloud = torch.from_numpy(point_cloud).float()
        
        # 如果是单个点云，添加batch维度
        if point_cloud.dim() == 2:
            point_cloud = point_cloud.unsqueeze(0)
        
        # 移到设备
        point_cloud = point_cloud.to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(point_cloud)
        
        # 返回CPU上的numpy数组
        return features.cpu().numpy()


# =====================================
# 第2部分：Gallery特征数据库
# =====================================

class GalleryDatabase:
    """Gallery特征数据库管理"""
    
    def __init__(self, gallery_dir='retrieval_data/gallery'):
        """
        Args:
            gallery_dir: Gallery数据目录
        """
        self.gallery_dir = Path(gallery_dir)
        self.features = None
        self.labels = None
        self.file_paths = []
        
        # 加载标签信息
        with open('retrieval_data/metadata/labels.pkl', 'rb') as f:
            self.label_info = pickle.load(f)
    
    def build_database(self, feature_extractor, save_path='gallery_features.npz'):
        """
        构建Gallery特征数据库
        
        Args:
            feature_extractor: 特征提取器
            save_path: 保存路径
        """
        print("\n" + "="*60)
        print("构建Gallery特征数据库")
        print("="*60)
        
        # 获取所有Gallery文件
        gallery_files = sorted(self.gallery_dir.glob('*.npz'))
        print(f"Gallery包含 {len(gallery_files)} 个样本")
        
        features_list = []
        labels_list = []
        
        # 批量处理以提高效率
        batch_size = 32
        for i in tqdm(range(0, len(gallery_files), batch_size), desc="提取Gallery特征"):
            batch_files = gallery_files[i:i+batch_size]
            batch_points = []
            
            # 加载批量点云
            for file_path in batch_files:
                data = np.load(file_path)
                points = data['points'].astype(np.float32)
                batch_points.append(points)
                self.file_paths.append(str(file_path))
            
            # 批量提取特征
            batch_points = np.array(batch_points)
            batch_features = feature_extractor.extract_features(batch_points)
            
            features_list.append(batch_features)
        
        # 合并所有特征
        self.features = np.vstack(features_list)
        self.labels = np.array(self.label_info['gallery_labels'])
        
        # 保存特征数据库
        np.savez(save_path,
                features=self.features,
                labels=self.labels,
                file_paths=self.file_paths)
        
        print(f"✓ Gallery特征数据库构建完成")
        print(f"  特征形状: {self.features.shape}")
        print(f"  保存到: {save_path}")
        
        return self.features
    
    def load_database(self, database_path='gallery_features.npz'):
        """加载已构建的特征数据库"""
        print(f"加载Gallery数据库: {database_path}")
        data = np.load(database_path)
        self.features = data['features']
        self.labels = data['labels']
        self.file_paths = data['file_paths'].tolist()
        print(f"✓ 加载完成: {self.features.shape[0]} 个特征")
        return self.features


# =====================================
# 第3部分：检索引擎
# =====================================

class RetrievalEngine:
    """检索引擎：执行实际的检索任务"""
    
    def __init__(self, gallery_database):
        """
        Args:
            gallery_database: Gallery数据库
        """
        self.gallery = gallery_database
        
        # 使用FAISS加速检索（如果安装了）
        self.use_faiss = False
        try:
            import faiss
            self.faiss = faiss
            self.use_faiss = True
            self._build_faiss_index()
            print("✓ 使用FAISS加速检索")
        except ImportError:
            print("⚠ FAISS未安装，使用numpy计算相似度")
    
    def _build_faiss_index(self):
        """构建FAISS索引以加速检索"""
        if not self.use_faiss:
            return
        
        # L2归一化的特征用内积就是余弦相似度
        d = self.gallery.features.shape[1]  # 特征维度
        self.index = self.faiss.IndexFlatIP(d)  # 内积索引
        
        # 归一化并添加到索引
        features_normalized = self.gallery.features.copy()
        self.faiss.normalize_L2(features_normalized)
        self.index.add(features_normalized)
    
    def search(self, query_features, k=5):
        """
        检索最相似的k个Gallery样本
        
        Args:
            query_features: 查询特征 (n_queries, feature_dim)
            k: 返回Top-K个结果
        
        Returns:
            similarities: 相似度分数 (n_queries, k)
            indices: Gallery中的索引 (n_queries, k)
        """
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        
        if self.use_faiss:
            # 使用FAISS检索
            query_normalized = query_features.copy()
            self.faiss.normalize_L2(query_normalized)
            similarities, indices = self.index.search(query_normalized, k)
        else:
            # 使用numpy计算余弦相似度
            # 归一化
            query_norm = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
            gallery_norm = self.gallery.features / np.linalg.norm(self.gallery.features, axis=1, keepdims=True)
            
            # 计算相似度矩阵
            sim_matrix = np.dot(query_norm, gallery_norm.T)
            
            # 获取Top-K
            indices = np.argsort(-sim_matrix, axis=1)[:, :k]
            similarities = np.take_along_axis(sim_matrix, indices, axis=1)
        
        return similarities, indices
    
    def search_single(self, query_feature, k=5):
        """检索单个查询"""
        similarities, indices = self.search(query_feature.reshape(1, -1), k)
        return similarities[0], indices[0]


# =====================================
# 第4部分：性能评估
# =====================================

class RetrievalEvaluator:
    """评估检索系统性能"""
    
    def __init__(self, retrieval_engine, query_dir='retrieval_data/query'):
        """
        Args:
            retrieval_engine: 检索引擎
            query_dir: Query数据目录
        """
        self.engine = retrieval_engine
        self.query_dir = Path(query_dir)
        
        # 加载Query标签
        with open('retrieval_data/metadata/labels.pkl', 'rb') as f:
            label_info = pickle.load(f)
            # 确保转换为numpy数组！
            self.query_labels = np.array(label_info['query_labels'])
            self.query_types = label_info['query_types']
    
    def evaluate(self, feature_extractor, save_results=True):
        """
        评估检索性能
        
        Args:
            feature_extractor: 特征提取器
            save_results: 是否保存结果
        """
        print("\n" + "="*60)
        print("评估检索性能")
        print("="*60)
        
        # 获取所有Query文件
        query_files = sorted(self.query_dir.glob('*.npz'))
        print(f"Query集包含 {len(query_files)} 个样本")
        
        # 提取Query特征
        print("提取Query特征...")
        query_features = []
        for file_path in tqdm(query_files):
            data = np.load(file_path)
            points = data['points'].astype(np.float32)
            feature = feature_extractor.extract_features(points)
            query_features.append(feature[0])
        
        query_features = np.array(query_features)
        
        # 执行检索
        print("执行检索...")
        k_values = [1, 5, 10, 20]
        similarities, indices = self.engine.search(query_features, k=max(k_values))
        
        # 计算指标
        results = self._compute_metrics(indices, k_values)
        
        # 分别评估seen和unseen
        seen_indices = [i for i, t in enumerate(self.query_types) if t == 'seen']
        unseen_indices = [i for i, t in enumerate(self.query_types) if t == 'unseen']
        
        if seen_indices:
            results['seen'] = self._compute_metrics(
                indices[seen_indices], k_values, self.query_labels[seen_indices]
            )
        
        if unseen_indices:
            results['unseen'] = self._compute_metrics(
                indices[unseen_indices], k_values, self.query_labels[unseen_indices]
            )
        
        # 打印结果
        self._print_results(results)
        
        # 保存详细结果
        if save_results:
            self._save_results(results, indices, similarities, query_files)
        
        return results
    
    def _compute_metrics(self, indices, k_values, labels=None):
        """计算检索指标"""
        if labels is None:
            labels = self.query_labels
        
        metrics = {}
        n_queries = len(indices)
        
        for k in k_values:
            top_k_indices = indices[:, :k]
            top_k_labels = self.engine.gallery.labels[top_k_indices]
            
            # Rank-K准确率：Top-K中是否包含正确答案
            correct = 0
            for i in range(n_queries):
                query_label = labels[i]
                if query_label in top_k_labels[i]:
                    correct += 1
            
            metrics[f'rank_{k}'] = correct / n_queries
        
        # 计算mAP (Mean Average Precision)
        ap_scores = []
        for i in range(n_queries):
            query_label = labels[i]
            retrieved_labels = self.engine.gallery.labels[indices[i]]
            
            # 计算AP
            precisions = []
            relevant_count = 0
            for j, label in enumerate(retrieved_labels[:20]):  # 只看前20个
                if label == query_label:
                    relevant_count += 1
                    precisions.append(relevant_count / (j + 1))
            
            if precisions:
                ap_scores.append(np.mean(precisions))
            else:
                ap_scores.append(0)
        
        metrics['mAP'] = np.mean(ap_scores)
        
        return metrics
    
    def _print_results(self, results):
        """打印评估结果"""
        print("\n" + "="*60)
        print("检索性能评估结果")
        print("="*60)
        
        print("\n整体性能:")
        print(f"  Rank-1 准确率: {results['rank_1']:.2%}")
        print(f"  Rank-5 准确率: {results['rank_5']:.2%}")
        print(f"  Rank-10 准确率: {results['rank_10']:.2%}")
        print(f"  Rank-20 准确率: {results['rank_20']:.2%}")
        print(f"  mAP: {results['mAP']:.4f}")
        
        if 'seen' in results:
            print("\nSeen身份（训练时见过）:")
            print(f"  Rank-1: {results['seen']['rank_1']:.2%}")
            print(f"  Rank-5: {results['seen']['rank_5']:.2%}")
            print(f"  mAP: {results['seen']['mAP']:.4f}")
        
        if 'unseen' in results:
            print("\nUnseen身份（完全陌生）:")
            print(f"  Rank-1: {results['unseen']['rank_1']:.2%}")
            print(f"  Rank-5: {results['unseen']['rank_5']:.2%}")
            print(f"  mAP: {results['unseen']['mAP']:.4f}")
    
    def _save_results(self, metrics, indices, similarities, query_files):
        """保存详细结果"""
        results = {
            'metrics': metrics,
            'retrieval_results': []
        }
        
        # 确保标签是列表（JSON序列化需要）
        query_labels_list = self.query_labels.tolist() if isinstance(self.query_labels, np.ndarray) else self.query_labels
        
        # 保存每个Query的检索结果
        for i in range(len(query_files)):
            query_result = {
                'query_file': str(query_files[i]),
                'query_label': int(query_labels_list[i]),
                'query_type': self.query_types[i],
                'top_5_indices': indices[i, :5].tolist(),
                'top_5_labels': self.engine.gallery.labels[indices[i, :5]].tolist(),
                'top_5_similarities': similarities[i, :5].tolist(),
                'top_5_files': [self.engine.gallery.file_paths[j] for j in indices[i, :5]]
            }
            results['retrieval_results'].append(query_result)
        
        # 保存JSON
        with open('retrieval_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ 详细结果保存到: retrieval_results.json")


# =====================================
# 第5部分：可视化分析
# =====================================

class RetrievalVisualizer:
    """可视化检索结果"""
    
    @staticmethod
    def plot_retrieval_examples(results_file='retrieval_results.json', n_examples=5):
        """
        可视化检索示例
        
        Args:
            results_file: 结果文件
            n_examples: 显示几个例子
        """
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # 找成功和失败的例子
        successes = []
        failures = []
        
        for r in results['retrieval_results']:
            if r['query_label'] in r['top_5_labels'][:1]:  # Top-1正确
                successes.append(r)
            else:
                failures.append(r)
        
        print(f"\n成功案例 (Top-1正确): {len(successes)}")
        print(f"失败案例 (Top-1错误): {len(failures)}")
        
        # 显示例子
        print("\n" + "="*40)
        print("成功案例示例:")
        print("="*40)
        for r in successes[:n_examples]:
            print(f"\nQuery: {Path(r['query_file']).name}")
            print(f"  真实标签: {r['query_label']}")
            print(f"  Top-5预测: {r['top_5_labels']}")
            print(f"  相似度: {[f'{s:.3f}' for s in r['top_5_similarities']]}")
        
        print("\n" + "="*40)
        print("失败案例示例:")
        print("="*40)
        for r in failures[:n_examples]:
            print(f"\nQuery: {Path(r['query_file']).name}")
            print(f"  真实标签: {r['query_label']}")
            print(f"  Top-5预测: {r['top_5_labels']}")
            print(f"  相似度: {[f'{s:.3f}' for s in r['top_5_similarities']]}")
    
    @staticmethod
    def plot_performance_comparison(results):
        """绘制性能对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Rank-K曲线
        k_values = [1, 5, 10, 20]
        overall_acc = [results[f'rank_{k}'] for k in k_values]
        
        ax = axes[0]
        ax.plot(k_values, overall_acc, 'o-', label='Overall', linewidth=2)
        
        if 'seen' in results:
            seen_acc = [results['seen'][f'rank_{k}'] for k in k_values]
            ax.plot(k_values, seen_acc, 's--', label='Seen', linewidth=2)
        
        if 'unseen' in results:
            unseen_acc = [results['unseen'][f'rank_{k}'] for k in k_values]
            ax.plot(k_values, unseen_acc, '^:', label='Unseen', linewidth=2)
        
        ax.set_xlabel('K')
        ax.set_ylabel('Rank-K Accuracy')
        ax.set_title('Retrieval Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # mAP对比
        ax = axes[1]
        categories = ['Overall']
        map_scores = [results['mAP']]
        
        if 'seen' in results:
            categories.append('Seen')
            map_scores.append(results['seen']['mAP'])
        
        if 'unseen' in results:
            categories.append('Unseen')
            map_scores.append(results['unseen']['mAP'])
        
        bars = ax.bar(categories, map_scores, color=['blue', 'green', 'orange'])
        ax.set_ylabel('mAP Score')
        ax.set_title('Mean Average Precision')
        ax.set_ylim([0, 1])
        
        # 添加数值标签
        for bar, score in zip(bars, map_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('retrieval_performance.png', dpi=150)
        print(f"\n✓ 性能图表保存到: retrieval_performance.png")
        plt.show()


# =====================================
# 第6部分：主函数
# =====================================

def main():
    """主流程：构建检索系统并评估"""
    
    print("="*60)
    print("3D人脸检索系统")
    print("="*60)
    
    # 1. 加载特征提取器
    print("\n[步骤1] 加载特征提取器")
    extractor = FeatureExtractor(
        model_path='face_feature_extractor_final.pth',
        device='cuda'
    )
    
    # 2. 构建Gallery数据库
    print("\n[步骤2] 构建Gallery数据库")
    gallery = GalleryDatabase('retrieval_data/gallery')
    
    # 检查是否已有特征数据库
    if Path('gallery_features.npz').exists():
        print("发现已有特征数据库，直接加载")
        gallery.load_database('gallery_features.npz')
    else:
        print("构建新的特征数据库")
        gallery.build_database(extractor, 'gallery_features.npz')
    
    # 3. 创建检索引擎
    print("\n[步骤3] 初始化检索引擎")
    engine = RetrievalEngine(gallery)
    
    # 4. 评估性能
    print("\n[步骤4] 评估检索性能")
    evaluator = RetrievalEvaluator(engine, 'retrieval_data/query')
    results = evaluator.evaluate(extractor, save_results=True)
    
    # 5. 可视化结果
    print("\n[步骤5] 可视化结果")
    visualizer = RetrievalVisualizer()
    visualizer.plot_retrieval_examples('retrieval_results.json', n_examples=3)
    visualizer.plot_performance_comparison(results)
    
    # 6. 交互式检索演示
    print("\n" + "="*60)
    print("交互式检索演示")
    print("="*60)
    
    # 随机选一个Query作为演示
    demo_query_file = list(Path('retrieval_data/query').glob('*.npz'))[0]
    print(f"演示Query: {demo_query_file.name}")
    
    # 加载并提取特征
    data = np.load(demo_query_file)
    query_points = data['points'].astype(np.float32)
    query_feature = extractor.extract_features(query_points)
    
    # 检索
    start_time = time.time()
    similarities, indices = engine.search_single(query_feature[0], k=10)
    retrieval_time = time.time() - start_time
    
    print(f"\n检索耗时: {retrieval_time*1000:.2f}ms")
    print(f"\nTop-10 检索结果:")
    print("-"*50)
    for i, (sim, idx) in enumerate(zip(similarities, indices)):
        gallery_file = Path(gallery.file_paths[idx]).name
        gallery_label = gallery.labels[idx]
        print(f"  {i+1}. {gallery_file}")
        print(f"     标签: {gallery_label}, 相似度: {sim:.4f}")
    
    print("\n" + "="*60)
    print("✓ 检索系统构建完成！")
    print("="*60)
    print("\n下一步：")
    print("1. 使用XAI分析检索结果")
    print("2. 解释为什么某些人脸被认为相似")
    print("3. 识别关键的面部特征区域")


if __name__ == "__main__":
    main()