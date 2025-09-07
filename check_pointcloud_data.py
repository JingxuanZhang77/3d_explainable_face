from visualizer import visualize_batch
from pathlib import Path

# 获取文件夹里所有NPZ文件
npz_files = list(Path('/home/jz97/3d_face_repo/obj_data_1030_pointcloud').glob('*.npz'))  # ← 改这里！

# 批量可视化（默认处理前5个）
visualize_batch(npz_files, output_dir='检查结果', max_samples=10)  # 可以改数量