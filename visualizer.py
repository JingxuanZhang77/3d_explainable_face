"""
点云可视化工具集 - 适用于无GUI环境
提供多种可视化方案，包括保存到文件和Web可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

class PointCloudVisualizer:
    """点云可视化工具类"""
    
    @staticmethod
    def save_matplotlib_views(points, save_path='point_cloud_views.png', 
                            colors=None, title="Point Cloud Views"):
        """
        使用Matplotlib保存多视角投影图
        适用于：服务器环境，生成静态图像
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 如果没有颜色，使用Z坐标着色
        if colors is None:
            colors = points[:, 2]
        
        # 3D视图
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                              c=colors, s=0.5, alpha=0.6, cmap='viridis')
        ax1.set_title('3D View')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # 正面视图 (XY)
        ax2 = fig.add_subplot(2, 3, 2)
        scatter2 = ax2.scatter(points[:, 0], points[:, 1], 
                              c=colors, s=0.5, alpha=0.6, cmap='viridis')
        ax2.set_title('Front View (XY)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_aspect('equal')
        
        # 侧面视图 (YZ)
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.scatter(points[:, 1], points[:, 2], 
                   c=colors, s=0.5, alpha=0.6, cmap='viridis')
        ax3.set_title('Side View (YZ)')
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')
        ax3.set_aspect('equal')
        
        # 顶部视图 (XZ)
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.scatter(points[:, 0], points[:, 2], 
                   c=colors, s=0.5, alpha=0.6, cmap='viridis')
        ax4.set_title('Top View (XZ)')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Z')
        ax4.set_aspect('equal')
        
        # 点云分布直方图
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.hist(points[:, 0], bins=50, alpha=0.5, label='X', color='red')
        ax5.hist(points[:, 1], bins=50, alpha=0.5, label='Y', color='green')
        ax5.hist(points[:, 2], bins=50, alpha=0.5, label='Z', color='blue')
        ax5.set_title('Coordinate Distribution')
        ax5.set_xlabel('Value')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        
        # 统计信息
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        stats_text = f"""
        Point Cloud Statistics:
        
        Number of points: {len(points)}
        
        X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]
        Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]
        Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]
        
        Mean: [{points[:, 0].mean():.3f}, {points[:, 1].mean():.3f}, {points[:, 2].mean():.3f}]
        Std:  [{points[:, 0].std():.3f}, {points[:, 1].std():.3f}, {points[:, 2].std():.3f}]
        """
        ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
        ax6.set_title('Statistics')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 可视化已保存到: {save_path}")
        return save_path
    
    @staticmethod
    def create_plotly_interactive(points, colors=None, save_html=True, 
                                 output_file='point_cloud_interactive.html'):
        """
        创建交互式3D可视化（HTML文件）
        适用于：需要交互式探索，可以在浏览器中打开
        """
        if colors is None:
            # 使用高度（Z轴）作为颜色
            colors = points[:, 2]
        
        # 创建3D散点图
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Z Value"),
                opacity=0.8
            )
        )])
        
        # 设置布局
        fig.update_layout(
            title="3D Point Cloud Visualization",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',  # 保持比例
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800,
            showlegend=False
        )
        
        if save_html:
            fig.write_html(output_file)
            print(f"✓ 交互式可视化已保存到: {output_file}")
            print(f"  请在浏览器中打开查看（支持旋转、缩放）")
        
        return fig
    
    @staticmethod
    def create_comparison_plot(points_list, labels=None, save_path='comparison.html'):
        """
        比较多个点云
        适用于：对比处理前后的点云，或不同人脸的点云
        """
        if labels is None:
            labels = [f"Point Cloud {i+1}" for i in range(len(points_list))]
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1  # 使用不同颜色
        
        for i, (points, label) in enumerate(zip(points_list, labels)):
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                name=label,
                marker=dict(
                    size=2,
                    color=colors[i % len(colors)],
                    opacity=0.6
                )
            ))
        
        fig.update_layout(
            title="Point Cloud Comparison",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=1200,
            height=800,
            showlegend=True
        )
        
        fig.write_html(save_path)
        print(f"✓ 对比可视化已保存到: {save_path}")
        return fig
    
    @staticmethod
    def export_for_meshlab(points, normals=None, colors=None, 
                          output_file='point_cloud.ply'):
        """
        导出为PLY格式，可以用MeshLab或CloudCompare打开
        适用于：需要专业3D软件查看
        """
        import struct
        
        num_points = len(points)
        
        # 准备颜色（如果没有提供，使用灰色）
        if colors is None:
            colors = np.ones((num_points, 3)) * 0.7
        elif len(colors.shape) == 1:
            # 如果是1D颜色（如标量值），转换为RGB
            cmap = plt.cm.viridis
            colors = cmap(colors / colors.max())[:, :3]
        
        # 写入PLY文件
        with open(output_file, 'wb') as f:
            # 写入头部
            header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z"""
            
            if normals is not None:
                header += """
property float nx
property float ny
property float nz"""
            
            header += """
property uchar red
property uchar green
property uchar blue
end_header
"""
            f.write(header.encode('ascii'))
            
            # 写入数据
            for i in range(num_points):
                # 位置
                f.write(struct.pack('fff', 
                                   points[i, 0], 
                                   points[i, 1], 
                                   points[i, 2]))
                
                # 法向量（如果有）
                if normals is not None:
                    f.write(struct.pack('fff',
                                       normals[i, 0],
                                       normals[i, 1],
                                       normals[i, 2]))
                
                # 颜色
                color_rgb = (colors[i] * 255).astype(np.uint8)
                f.write(struct.pack('BBB',
                                   color_rgb[0],
                                   color_rgb[1],
                                   color_rgb[2]))
        
        print(f"✓ PLY文件已保存到: {output_file}")
        print(f"  可以使用MeshLab, CloudCompare或Blender打开")
        return output_file
    
    @staticmethod
    def create_depth_maps(points, save_path='depth_maps.png'):
        """
        生成深度图（从不同角度）
        适用于：2D CNN处理或快速预览
        """
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # 正面深度图
        axes[0, 0].hexbin(points[:, 0], points[:, 1], C=points[:, 2], 
                         gridsize=50, cmap='viridis')
        axes[0, 0].set_title('Front Depth Map')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        axes[0, 0].set_aspect('equal')
        
        # 侧面深度图
        axes[0, 1].hexbin(points[:, 1], points[:, 2], C=points[:, 0], 
                         gridsize=50, cmap='viridis')
        axes[0, 1].set_title('Side Depth Map')
        axes[0, 1].set_xlabel('Y')
        axes[0, 1].set_ylabel('Z')
        axes[0, 1].set_aspect('equal')
        
        # 顶部深度图
        axes[1, 0].hexbin(points[:, 0], points[:, 2], C=points[:, 1], 
                         gridsize=50, cmap='viridis')
        axes[1, 0].set_title('Top Depth Map')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Z')
        axes[1, 0].set_aspect('equal')
        
        # 点密度图
        from scipy.stats import gaussian_kde
        xy = points[:, :2].T
        z = gaussian_kde(xy)(xy)
        axes[1, 1].scatter(points[:, 0], points[:, 1], c=z, s=1, cmap='hot')
        axes[1, 1].set_title('Point Density Map')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Y')
        axes[1, 1].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 深度图已保存到: {save_path}")
        return save_path
    
    @staticmethod
    def quick_stats(points, save_json=True, output_file='point_cloud_stats.json'):
        """
        快速统计分析
        """
        stats = {
            'num_points': int(len(points)),
            'bbox': {
                'min': points.min(axis=0).tolist(),
                'max': points.max(axis=0).tolist(),
                'center': points.mean(axis=0).tolist()
            },
            'statistics': {
                'mean': points.mean(axis=0).tolist(),
                'std': points.std(axis=0).tolist(),
                'median': np.median(points, axis=0).tolist()
            },
            'ranges': {
                'x_range': float(points[:, 0].max() - points[:, 0].min()),
                'y_range': float(points[:, 1].max() - points[:, 1].min()),
                'z_range': float(points[:, 2].max() - points[:, 2].min())
            },
            'spread': float(np.sqrt(np.sum(points.var(axis=0)))),
            'centroid_distance': {
                'mean': float(np.mean(np.linalg.norm(points - points.mean(axis=0), axis=1))),
                'max': float(np.max(np.linalg.norm(points - points.mean(axis=0), axis=1)))
            }
        }
        
        if save_json:
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"✓ 统计信息已保存到: {output_file}")
        
        # 打印统计信息
        print("\n点云统计信息:")
        print(f"  点数: {stats['num_points']}")
        print(f"  边界框: {stats['bbox']['min']} 到 {stats['bbox']['max']}")
        print(f"  中心: {stats['bbox']['center']}")
        print(f"  标准差: {stats['statistics']['std']}")
        print(f"  扩散度: {stats['spread']:.4f}")
        
        return stats


def visualize_batch(npz_files, output_dir='visualizations', max_samples=5):
    """
    批量可视化多个点云文件
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    visualizer = PointCloudVisualizer()
    
    for i, npz_file in enumerate(npz_files[:max_samples]):
        print(f"\n处理 {i+1}/{min(len(npz_files), max_samples)}: {npz_file}")
        
        # 加载数据
        data = np.load(npz_file)
        points = data['points']
        normals = data.get('normals', None)
        
        # ================================================================= #
        # 在这里添加坐标变换代码！
        # ================================================================= #
        # 复制一份原始点云，避免修改原始数据
        points_corrected = points.copy()
        
        # points_corrected[:, 1] = points[:, 2]

# 新的 Z 轴 (深度/前后) = 原始的 Y 轴
        # points_corrected[:, 2] = points[:, 1]
        # （可选）如果觉得人脸是反的（左右镜像），可以取消下面这行注释
        # points_corrected[:, 0] = -points[:, 0] 

# （可选）如果觉得人脸是倒的（上下颠倒），可以取消下面这行注释
        points_corrected[:, 1] = -points_corrected[:, 1]
        
        # 如果有法向量，法向量也要进行同样的旋转
        if normals is not None:
            normals_corrected = normals.copy()
            normals_corrected[:, 1] = -normals[:, 2]
            normals_corrected[:, 2] = normals[:, 1]
        else:
            normals_corrected = None
            
        # ================================================================= #
        
        # 生成文件名前缀
        file_stem = Path(npz_file).stem
        
        # 1. 生成多视角静态图
        # 注意：现在使用修正后的点云 points_corrected
        visualizer.save_matplotlib_views(
            points_corrected, # <--- 使用修正后的数据
            save_path=output_path / f"{file_stem}_views.png"
        )
        
        # 2. 生成交互式HTML
        visualizer.create_plotly_interactive(
            points_corrected, # <--- 使用修正后的数据
            output_file=output_path / f"{file_stem}_interactive.html"
        )
        
        # 3. 生成深度图
        visualizer.create_depth_maps(
            points_corrected, # <--- 使用修正后的数据
            save_path=output_path / f"{file_stem}_depth.png"
        )
        
        # 4. 导出PLY（可选）
        visualizer.export_for_meshlab(
            points_corrected, # <--- 使用修正后的数据
            normals=normals_corrected, # <--- 使用修正后的法向量
            output_file=output_path / f"{file_stem}.ply"
        )
        
        # 5. 保存统计信息
        visualizer.quick_stats(
            points_corrected, # <--- 使用修正后的数据
            output_file=output_path / f"{file_stem}_stats.json"
        )
    
    print(f"\n✓ 所有可视化已保存到: {output_path}")

def compare_preprocessing_results(original_obj, processed_npz):
    """
    对比预处理前后的结果
    """
    import trimesh
    
    # 加载原始mesh
    mesh = trimesh.load(original_obj, force='mesh')
    original_points = mesh.vertices
    
    # 加载处理后的点云
    data = np.load(processed_npz)
    processed_points = data['points']
    
    visualizer = PointCloudVisualizer()
    
    # 创建对比可视化
    visualizer.create_comparison_plot(
        [original_points, processed_points],
        labels=['Original Mesh Vertices', 'Processed Point Cloud'],
        save_path='preprocessing_comparison.html'
    )
    
    print("✓ 对比可视化已生成")


# 使用示例
if __name__ == "__main__":
    # 示例：可视化单个点云
    print("=" * 50)
    print("点云可视化示例")
    print("=" * 50)
    
    # 生成示例数据（替换为您的实际数据）
    sample_points = np.random.randn(2048, 3) * 0.5
    
    visualizer = PointCloudVisualizer()
    
    # 1. 保存多视角图像（无需GUI）
    visualizer.save_matplotlib_views(sample_points, save_path='sample_views.png')
    
    # 2. 创建交互式HTML（可在浏览器查看）
    visualizer.create_plotly_interactive(sample_points, output_file='sample_3d.html')
    
    # 3. 生成深度图
    visualizer.create_depth_maps(sample_points, save_path='sample_depth.png')
    
    # 4. 导出为PLY格式
    visualizer.export_for_meshlab(sample_points, output_file='sample.ply')
    
    # 5. 输出统计信息
    visualizer.quick_stats(sample_points)
    
    print("\n" + "=" * 50)
    print("✓ 所有可视化文件已生成！")
    print("  - sample_views.png: 多视角静态图")
    print("  - sample_3d.html: 交互式3D视图（在浏览器打开）")
    print("  - sample_depth.png: 深度图")
    print("  - sample.ply: 3D模型文件（用MeshLab打开）")
    print("  - point_cloud_stats.json: 统计数据")
    print("=" * 50)