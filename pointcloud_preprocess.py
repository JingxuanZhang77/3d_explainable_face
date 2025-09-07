"""
3D人脸OBJ文件批量预处理脚本
将OBJ格式的3D人脸模型转换为标准化的点云数据
"""

import os
import numpy as np
import trimesh
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import json
import pickle
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class FacePointCloudPreprocessor:
    """3D人脸点云预处理器"""
    
    def __init__(self, 
                 num_points=2048,
                 use_normals=True,
                 align_faces=True,
                 normalize_scale=True):
        """
        参数:
            num_points: 每个点云采样的点数
            use_normals: 是否计算并保存法向量
            align_faces: 是否对齐人脸
            normalize_scale: 是否归一化尺度
        """
        self.num_points = num_points
        self.use_normals = use_normals
        self.align_faces = align_faces
        self.normalize_scale = normalize_scale
        self.preprocessing_stats = {}
        
    def load_obj(self, obj_path):
        """加载OBJ文件"""
        try:
            # 使用trimesh加载
            mesh = trimesh.load(obj_path, force='mesh', process=False)
            return mesh
        except Exception as e:
            print(f"Error loading {obj_path}: {e}")
            # 备用方案：使用Open3D
            try:
                mesh = o3d.io.read_triangle_mesh(str(obj_path))
                # 转换为trimesh格式
                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                return mesh
            except:
                return None
    
    def clean_mesh(self, mesh):
        """清理mesh：去除噪声、修复问题"""
        # 移除重复顶点
        mesh.merge_vertices()
        
        # 移除退化的面（面积为0的三角形）
        mesh.remove_degenerate_faces()
        
        # 移除未引用的顶点
        mesh.remove_unreferenced_vertices()
        
        # 填补小洞（可选）
        if len(mesh.vertices) > 100:  # 确保mesh有足够的顶点
            mesh.fill_holes()
        
        return mesh
    
    def sample_point_cloud(self, mesh, method='uniform'):
        """
        从mesh采样点云
        
        方法:
            - 'uniform': 均匀采样（推荐）
            - 'poisson': 泊松盘采样（更均匀但较慢）
            - 'surface': 表面随机采样
        """
        if method == 'uniform':
            # 均匀采样
            points, face_indices = trimesh.sample.sample_surface(
                mesh, self.num_points, seed=42
            )
            
        elif method == 'poisson':
            # 泊松盘采样（更均匀的分布）
            points, face_indices = trimesh.sample.sample_surface(
                mesh, self.num_points * 10, seed=42  # 过采样
            )
            # 使用最远点采样进行下采样
            points = self.farthest_point_sampling(points, self.num_points)
            
        else:  # surface
            points, face_indices = trimesh.sample.sample_surface_even(
                mesh, self.num_points, seed=42
            )
        
        # 计算法向量（如果需要）
        normals = None
        if self.use_normals and hasattr(mesh, 'face_normals'):
            # 获取采样点对应的面法向量
            if face_indices is not None and len(face_indices) == len(points):
                normals = mesh.face_normals[face_indices]
            else:
                # 备用方案：计算顶点法向量并插值
                mesh.vertex_normals  # 触发计算
                # 找最近的顶点
                from scipy.spatial import KDTree
                tree = KDTree(mesh.vertices)
                _, indices = tree.query(points)
                normals = mesh.vertex_normals[indices]
        
        return points, normals
    
    def farthest_point_sampling(self, points, n_samples):
        """最远点采样算法，获得更均匀的点分布"""
        n_points = points.shape[0]
        
        # 随机选择第一个点
        sampled_indices = [np.random.randint(n_points)]
        distances = np.full(n_points, np.inf)
        
        for _ in range(n_samples - 1):
            current_point = points[sampled_indices[-1]]
            
            # 更新到最近已选点的距离
            new_distances = np.linalg.norm(points - current_point, axis=1)
            distances = np.minimum(distances, new_distances)
            
            # 选择距离最远的点
            next_idx = np.argmax(distances)
            sampled_indices.append(next_idx)
        
        return points[sampled_indices]
    
    def align_face(self, points, normals=None):
        """对齐人脸到标准位置"""
        if not self.align_faces:
            return points, normals
        
        # 使用PCA找到主轴
        pca = PCA(n_components=3)
        pca.fit(points)
        
        # 旋转到主轴对齐
        points_aligned = pca.transform(points)
        
        # 确保人脸朝向一致（假设第一主成分是垂直方向）
        if np.mean(points_aligned[:, 0]) < 0:
            points_aligned[:, 0] *= -1
        
        if normals is not None:
            # 同样变换法向量
            normals = pca.transform(normals)
        
        return points_aligned, normals
    
    def normalize_points(self, points):
        """归一化点云到单位球"""
        if not self.normalize_scale:
            return points
        
        # 中心化
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # 缩放到单位球
        max_distance = np.max(np.linalg.norm(points, axis=1))
        points = points / max_distance
        
        return points
    
    def augment_point_cloud(self, points, normals=None, augmentation_params=None):
        """数据增强（可选）"""
        if augmentation_params is None:
            return points, normals
        
        augmented_points = points.copy()
        augmented_normals = normals.copy() if normals is not None else None
        
        # 随机旋转
        if 'rotation' in augmentation_params:
            angle_range = augmentation_params['rotation']
            angles = np.random.uniform(-angle_range, angle_range, 3)
            rotation_matrix = self.get_rotation_matrix(angles)
            augmented_points = augmented_points @ rotation_matrix.T
            if augmented_normals is not None:
                augmented_normals = augmented_normals @ rotation_matrix.T
        
        # 随机缩放
        if 'scale' in augmentation_params:
            scale_range = augmentation_params['scale']
            scale = np.random.uniform(1 - scale_range, 1 + scale_range)
            augmented_points *= scale
        
        # 添加噪声
        if 'noise' in augmentation_params:
            noise_level = augmentation_params['noise']
            noise = np.random.normal(0, noise_level, augmented_points.shape)
            augmented_points += noise
        
        # 随机丢弃点（dropout）
        if 'dropout' in augmentation_params:
            dropout_rate = augmentation_params['dropout']
            n_keep = int(len(augmented_points) * (1 - dropout_rate))
            keep_indices = np.random.choice(len(augmented_points), n_keep, replace=False)
            augmented_points = augmented_points[keep_indices]
            if augmented_normals is not None:
                augmented_normals = augmented_normals[keep_indices]
        
        return augmented_points, augmented_normals
    
    def get_rotation_matrix(self, angles):
        """生成3D旋转矩阵"""
        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles)
        
        # X轴旋转
        Rx = np.array([
            [1, 0, 0],
            [0, cos_vals[0], -sin_vals[0]],
            [0, sin_vals[0], cos_vals[0]]
        ])
        
        # Y轴旋转
        Ry = np.array([
            [cos_vals[1], 0, sin_vals[1]],
            [0, 1, 0],
            [-sin_vals[1], 0, cos_vals[1]]
        ])
        
        # Z轴旋转
        Rz = np.array([
            [cos_vals[2], -sin_vals[2], 0],
            [sin_vals[2], cos_vals[2], 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx
    
    def validate_point_cloud(self, points):
        """验证点云质量"""
        issues = []
        
        # 检查点数
        if len(points) != self.num_points:
            issues.append(f"Point count mismatch: {len(points)} vs {self.num_points}")
        
        # 检查NaN或Inf
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            issues.append("Contains NaN or Inf values")
        
        # 检查范围
        if self.normalize_scale:
            max_dist = np.max(np.linalg.norm(points, axis=1))
            if max_dist > 1.5:  # 允许一些容差
                issues.append(f"Points exceed unit sphere: max_dist={max_dist}")
        
        # 检查点云分布
        std = np.std(points, axis=0)
        if np.any(std < 0.01):
            issues.append(f"Degenerate distribution detected: std={std}")
        
        return len(issues) == 0, issues
    
    def process_single_obj(self, obj_path, save_debug_info=False):
        """处理单个OBJ文件"""
        result = {
            'success': False,
            'obj_path': str(obj_path),
            'error': None,
            'points': None,
            'normals': None,
            'metadata': {}
        }
        
        try:
            # 1. 加载mesh
            mesh = self.load_obj(obj_path)
            if mesh is None:
                result['error'] = "Failed to load mesh"
                return result
            
            # 记录原始信息
            result['metadata']['original_vertices'] = len(mesh.vertices)
            result['metadata']['original_faces'] = len(mesh.faces)
            
            # 2. 清理mesh
            mesh = self.clean_mesh(mesh)
            
            # 3. 采样点云
            points, normals = self.sample_point_cloud(mesh, method='uniform')
            
            # 4. 对齐人脸
            points, normals = self.align_face(points, normals)
            
            # 5. 归一化
            points = self.normalize_points(points)
            
            # 6. 验证
            is_valid, issues = self.validate_point_cloud(points)
            if not is_valid:
                result['error'] = f"Validation failed: {issues}"
                return result
            
            # 保存结果
            result['success'] = True
            result['points'] = points.astype(np.float32)
            if self.use_normals and normals is not None:
                result['normals'] = normals.astype(np.float32)
            
            # 保存调试信息
            if save_debug_info:
                result['metadata']['point_cloud_stats'] = {
                    'mean': points.mean(axis=0).tolist(),
                    'std': points.std(axis=0).tolist(),
                    'min': points.min(axis=0).tolist(),
                    'max': points.max(axis=0).tolist()
                }
            
        except Exception as e:
            result['error'] = str(e)
            print(f"Error processing {obj_path}: {e}")
        
        return result
    
    def process_batch(self, 
                     input_dir, 
                     output_dir,
                     output_format='npz',
                     save_metadata=True,
                     num_augmentations=0,
                     augmentation_params=None):
        """
        批量处理OBJ文件
        
        参数:
            input_dir: OBJ文件目录
            output_dir: 输出目录
            output_format: 'npz', 'npy', 'pkl', 'h5'
            save_metadata: 是否保存元数据
            num_augmentations: 每个样本生成的增强版本数
            augmentation_params: 增强参数
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有OBJ文件
        obj_files = list(input_path.glob('**/*.obj'))
        print(f"Found {len(obj_files)} OBJ files")
        
        # 处理结果统计
        processed_data = []
        failed_files = []
        metadata_all = {}
        
        # 批量处理
        for obj_file in tqdm(obj_files, desc="Processing OBJ files"):
            # 获取相对路径作为ID
            relative_path = obj_file.relative_to(input_path)
            file_id = str(relative_path).replace('.obj', '').replace(os.sep, '_')
            
            # 处理原始文件
            result = self.process_single_obj(obj_file)
            
            if result['success']:
                # 保存原始版本
                data_entry = {
                    'id': file_id,
                    'points': result['points'],
                    'normals': result.get('normals'),
                    'source': str(relative_path)
                }
                processed_data.append(data_entry)
                
                # 保存单个文件
                if output_format == 'npz':
                    output_file = output_path / f"{file_id}.npz"
                    save_data = {'points': result['points']}
                    if result.get('normals') is not None:
                        save_data['normals'] = result['normals']
                    np.savez_compressed(output_file, **save_data)
                
                # 生成增强版本
                if num_augmentations > 0 and augmentation_params:
                    for aug_idx in range(num_augmentations):
                        aug_points, aug_normals = self.augment_point_cloud(
                            result['points'],
                            result.get('normals'),
                            augmentation_params
                        )
                        
                        aug_entry = {
                            'id': f"{file_id}_aug{aug_idx}",
                            'points': aug_points,
                            'normals': aug_normals,
                            'source': str(relative_path),
                            'augmented': True
                        }
                        processed_data.append(aug_entry)
                
                # 保存元数据
                if save_metadata:
                    metadata_all[file_id] = result['metadata']
            else:
                failed_files.append({
                    'file': str(relative_path),
                    'error': result['error']
                })
        
        # 保存汇总数据
        if output_format == 'pkl':
            # 保存为pickle格式（包含所有数据）
            with open(output_path / 'all_point_clouds.pkl', 'wb') as f:
                pickle.dump(processed_data, f)
        
        elif output_format == 'h5':
            # 保存为HDF5格式（适合大数据集）
            import h5py
            with h5py.File(output_path / 'all_point_clouds.h5', 'w') as f:
                for entry in processed_data:
                    grp = f.create_group(entry['id'])
                    grp.create_dataset('points', data=entry['points'])
                    if entry.get('normals') is not None:
                        grp.create_dataset('normals', data=entry['normals'])
                    grp.attrs['source'] = entry['source']
        
        # 保存元数据
        if save_metadata:
            with open(output_path / 'metadata.json', 'w') as f:
                json.dump({
                    'preprocessing_params': {
                        'num_points': self.num_points,
                        'use_normals': self.use_normals,
                        'align_faces': self.align_faces,
                        'normalize_scale': self.normalize_scale
                    },
                    'statistics': {
                        'total_files': len(obj_files),
                        'successful': len(processed_data) - sum(1 for d in processed_data if d.get('augmented', False)),
                        'failed': len(failed_files),
                        'augmented_samples': sum(1 for d in processed_data if d.get('augmented', False))
                    },
                    'failed_files': failed_files,
                    'file_metadata': metadata_all
                }, f, indent=2)
        
        # 打印统计信息
        print(f"\n处理完成:")
        print(f"  成功: {len(processed_data)} 个点云")
        print(f"  失败: {len(failed_files)} 个文件")
        if failed_files:
            print(f"  失败文件列表已保存到 metadata.json")
        
        return processed_data, failed_files


def main():
    """主函数：批量处理示例"""
    
    # 配置参数
    config = {
        'input_dir': '/home/jz97/3d_face_repo/obj_data_1030',  # 修改为您的OBJ文件目录
        'output_dir': '/home/jz97/3d_face_repo/obj_data_1030_pointcloud',  # 输出目录
        'num_points': 2048,  # 每个点云的点数
        'use_normals': True,  # 是否保存法向量
        'align_faces': True,  # 是否对齐人脸
        'normalize_scale': True,  # 是否归一化到单位球
        'output_format': 'npz',  # 输出格式
        'save_metadata': True,  # 保存元数据
        'num_augmentations': 0,  # 数据增强数量（0表示不增强）
    }
    
    # 数据增强参数（如果需要）
    augmentation_params = {
        'rotation': np.pi / 36,  # ±5度旋转
        'scale': 0.05,  # ±5%缩放
        'noise': 0.001,  # 噪声级别
        # 'dropout': 0.1  # 随机丢弃10%的点
    }
    
    # 创建预处理器
    preprocessor = FacePointCloudPreprocessor(
        num_points=config['num_points'],
        use_normals=config['use_normals'],
        align_faces=config['align_faces'],
        normalize_scale=config['normalize_scale']
    )
    
    # 批量处理
    processed_data, failed_files = preprocessor.process_batch(
        input_dir=config['input_dir'],
        output_dir=config['output_dir'],
        output_format=config['output_format'],
        save_metadata=config['save_metadata'],
        num_augmentations=config['num_augmentations'],
        augmentation_params=augmentation_params if config['num_augmentations'] > 0 else None
    )
    
    print("\n预处理完成！")
    
    # 可选：快速可视化检查
    if processed_data:
        print("\n可视化第一个处理后的点云...")
        visualize_point_cloud(processed_data[0]['points'])


def visualize_point_cloud(points, colors=None):
    """使用Open3D可视化点云（用于调试）"""
    import open3d as o3d
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # 默认颜色
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    # 可视化
    o3d.visualization.draw_geometries([pcd],
                                    window_name="Point Cloud Visualization",
                                    width=800,
                                    height=600)


if __name__ == "__main__":
    main()