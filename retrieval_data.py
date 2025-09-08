"""
人脸检索任务的数据重组织（覆盖旧产物，保证 metadata 与目录严格对齐）
"""

import numpy as np
from pathlib import Path
import json
import shutil
import random
from tqdm import tqdm
from collections import defaultdict
import pickle
from datetime import datetime

def _empty_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    for p in d.iterdir():
        if p.is_file() or p.is_symlink():
            p.unlink()
        else:
            shutil.rmtree(p)

class RetrievalDataOrganizer:
    """将增强数据重组织为检索任务格式"""
    
    def __init__(self, augmented_dir, output_dir, seed=42, reset_output=True):
        """
        Args:
            augmented_dir: 增强数据目录（包含train/train_augmented/val/test）
            output_dir: 输出目录
            seed: 随机种子
            reset_output: 是否在运行前清空 train/gallery/query 目录
        """
        self.augmented_dir = Path(augmented_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reset_output = reset_output
        
        random.seed(seed)
        np.random.seed(seed)
        
        # 创建输出子目录
        self.dirs = {
            'train': self.output_dir / 'train',           # 训练特征提取器
            'gallery': self.output_dir / 'gallery',       # 数据库（检索库）
            'query': self.output_dir / 'query',           # 查询集
            'metadata': self.output_dir / 'metadata'      # 元数据
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def analyze_current_data(self):
        """分析当前的数据结构"""
        print("="*60)
        print("分析现有数据结构")
        print("="*60)
        
        data_info = {}
        
        # 训练集（原始）
        train_identities = set()
        for f in (self.augmented_dir / 'train').glob('*.npz'):
            stem = f.stem
            if '_aug' in stem:  # 容错：防止误放增强到 train
                continue
            train_identities.add(stem)
        
        # 训练增强
        train_aug_files = defaultdict(list)
        for f in (self.augmented_dir / 'train_augmented').glob('*.npz'):
            stem = f.stem
            if '_aug' in stem:
                identity = stem.rsplit('_aug', 1)[0]
            else:
                identity = stem
            train_aug_files[identity].append(f)
        
        data_info['train'] = {
            'identities': train_identities,
            'n_identities': len(train_identities),
            'augmented_files': train_aug_files,
            'total_samples': len(train_identities) + sum(len(v) for v in train_aug_files.values())
        }
        
        # 验证集
        val_identities = set(f.stem for f in (self.augmented_dir / 'val').glob('*.npz'))
        data_info['val'] = {'identities': val_identities, 'n_identities': len(val_identities)}
        
        # 测试集
        test_identities = set(f.stem for f in (self.augmented_dir / 'test').glob('*.npz'))
        data_info['test'] = {'identities': test_identities, 'n_identities': len(test_identities)}
        
        print(f"训练集: {data_info['train']['n_identities']} 身份, 样本(含增强)≈ {data_info['train']['total_samples']}")
        print(f"验证集: {data_info['val']['n_identities']} 身份")
        print(f"测试集: {data_info['test']['n_identities']} 身份")
        return data_info
    
    def create_retrieval_splits(self, data_info):
        """
        创建检索任务的数据划分
        策略：
        - Gallery：训练身份的一部分 + 验证集
        - Query：训练的（这里你选的是 gallery 的训练子集） + 测试集
        """
        print("\n" + "="*60)
        print("创建检索任务数据划分")
        print("="*60)
        
        train_ids = list(data_info['train']['identities'])
        val_ids = list(data_info['val']['identities'])
        test_ids = list(data_info['test']['identities'])
        
        random.shuffle(train_ids)
        n_train_for_gallery = int(len(train_ids) * 0.7)
        train_gallery_ids = train_ids[:n_train_for_gallery]
        train_query_ids = train_ids[n_train_for_gallery:]  # 仅用于打印参考

        splits = {
            'train': {
                'identities': set(train_ids),
                'description': '用于训练特征提取器'
            },
            'gallery': {
                'identities': set(train_gallery_ids) | set(val_ids),  # 部分训练 + 验证
                'description': '检索数据库'
            },
            'query': {
                # 你当前的需求：用“进入 gallery 的训练子集” + 全部测试身份
                'identities': set(train_gallery_ids) | set(test_ids),
                'description': '查询测试集（已见训练子集 + 未见测试）'
            }
        }
        
        print(f"\nGallery: {len(splits['gallery']['identities'])} 身份 "
              f"(train_part={len(train_gallery_ids)}, val={len(val_ids)})")
        print(f"Query:   {len(splits['query']['identities'])} 身份 "
              f"(train_part={len(train_gallery_ids)}, test={len(test_ids)})")
        print(f"提示：train 中未进 gallery 的 {len(train_query_ids)} 身份未进入 query（与你当前设定一致）")
        return splits
    
    def copy_files_for_split(self, splits, data_info):
        """根据划分复制文件到新目录（Gallery/Query 严格仅用原始数据）"""
        print("\n" + "="*60)
        print("复制文件到新目录结构（原始样本-only）")
        print("="*60)

        file_mapping = {}

        # ---------- 0) 清空旧的 gallery / query，避免残留 ----------
        for sub in ['gallery', 'query']:
            for p in self.dirs[sub].glob('*'):
                if p.is_file() or p.is_symlink():
                    p.unlink()
                else:
                    shutil.rmtree(p)

        # ---------- 1) 训练集（保持你原有逻辑：原始 + 增强） ----------
        print("\n处理训练集...")
        train_files = []
        for identity in tqdm(sorted(splits['train']['identities']), desc="Train"):
            # 原始
            raw = self.augmented_dir / 'train' / f"{identity}.npz"
            if raw.exists():
                dst = self.dirs['train'] / raw.name
                shutil.copy2(raw, dst)
                train_files.append(str(dst))
            # 增强（训练用，不影响你的要求）
            for aug in data_info['train']['augmented_files'].get(identity, []):
                dst = self.dirs['train'] / aug.name
                shutil.copy2(aug, dst)
                train_files.append(str(dst))
        file_mapping['train'] = train_files
        print(f"  训练复制 {len(train_files)} 个样本")

        # ---------- 2) Gallery（仅原始：优先 train，其次 val；每身份 1 个文件） ----------
        print("\n处理 Gallery（原始-only）...")
        gallery_files, gallery_labels, label_map = [], [], {}
        current_label = 0

        for identity in tqdm(sorted(splits['gallery']['identities']), desc="Gallery"):
            # 只找原始样本，不使用任何增强
            src = None
            for split_name in ['train', 'val']:
                cand = self.augmented_dir / split_name / f"{identity}.npz"
                if cand.exists():
                    src = cand
                    break
            if src is None:
                print(f"⚠️ 跳过 {identity}：train/val 中未找到原始样本")
                continue

            dst = self.dirs['gallery'] / f"{identity}.npz"
            shutil.copy2(src, dst)

            label_map[identity] = current_label
            gallery_files.append(str(dst))
            gallery_labels.append(current_label)
            current_label += 1

        file_mapping['gallery'] = {
            'files': gallery_files,
            'labels': gallery_labels,
            'label_map': label_map
        }
        print(f"  Gallery 样本 {len(gallery_files)}，身份 {len(label_map)}")

        # ---------- 3) Query（仅原始：
        #     已见=来自 train_gallery_ids → train/identity.npz
        #     未见=测试身份 → test/identity.npz
        # ）----------
        print("\n处理 Query（原始-only）...")
        query_files, query_labels, query_types = [], [], []

        # 在你的 create_retrieval_splits 里，query 身份 = train_gallery_ids ∪ test_ids
        # 因此：只会从 train/ 或 test/ 拿原始样本
        for identity in tqdm(sorted(splits['query']['identities']), desc="Query"):
            if identity in data_info['train']['identities']:
                # 已见（来自训练子集）
                src = self.augmented_dir / 'train' / f"{identity}.npz"
                qtype = 'seen'
            else:
                # 未见（测试）
                src = self.augmented_dir / 'test' / f"{identity}.npz"
                qtype = 'unseen'

            if not src.exists():
                print(f"⚠️ 跳过 Query 身份 {identity}：未找到原始样本（{qtype}）")
                continue

            dst = self.dirs['query'] / f"{identity}.npz"
            shutil.copy2(src, dst)

            query_files.append(str(dst))
            query_labels.append(label_map.get(identity, -1))  # 不在 gallery 的测试身份 → -1
            query_types.append(qtype)

        file_mapping['query'] = {
            'files': query_files,
            'labels': query_labels,
            'types': query_types,
            'n_seen': sum(t == 'seen' for t in query_types),
            'n_unseen': sum(t == 'unseen' for t in query_types)
        }
        print(f"  Query 样本 {len(query_files)}（seen {file_mapping['query']['n_seen']} | unseen {file_mapping['query']['n_unseen']}）")

        return file_mapping

    def create_metadata(self, splits, file_mapping):
        """创建元数据文件（与实际文件严格一致）"""
        print("\n" + "="*60)
        print("生成元数据")
        print("="*60)

        self.dirs['metadata'].mkdir(parents=True, exist_ok=True)
        metadata = {
            'task': 'face_retrieval',
            'creation_time': datetime.now().isoformat(timespec='seconds'),
            'data_splits': {
                'train': {
                    'n_identities': len(splits['train']['identities']),
                    'n_samples': len(file_mapping['train']),
                    'description': '用于训练特征提取器'
                },
                'gallery': {
                    'n_identities': len(file_mapping['gallery']['label_map']),
                    'n_samples': len(file_mapping['gallery']['files']),
                    'description': '检索数据库'
                },
                'query': {
                    'n_samples': len(file_mapping['query']['files']),
                    'n_seen': file_mapping['query']['n_seen'],
                    'n_unseen': file_mapping['query']['n_unseen'],
                    'description': '查询测试集（已见训练子集 + 未见测试）'
                }
            }
        }
        with open(self.dirs['metadata'] / 'retrieval_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 写清单（仅文件名，供评测严格按顺序读）
        for split_name in ['train', 'gallery', 'query']:
            if split_name == 'train':
                file_list = [Path(p).name for p in file_mapping['train']]
            else:
                file_list = [Path(p).name for p in file_mapping[split_name]['files']]
            with open(self.dirs['metadata'] / f'{split_name}_files.txt', 'w') as f:
                for name in file_list:
                    f.write(name + '\n')

        # 写标签映射
        with open(self.dirs['metadata'] / 'labels.pkl', 'wb') as f:
            pickle.dump({
                'gallery_labels': file_mapping['gallery']['labels'],
                'gallery_label_map': file_mapping['gallery']['label_map'],
                'query_labels': file_mapping['query']['labels'],
                'query_types': file_mapping['query']['types']
            }, f)
        print(f"元数据已保存到: {self.dirs['metadata']}")
        return metadata
    
    def create_train_pairs_for_metric_learning(self, file_mapping):
        """为度量学习创建训练三元组（与原逻辑一致）"""
        print("\n创建度量学习训练数据...")
        identity_files = defaultdict(list)
        for file_path in file_mapping['train']:
            name = Path(file_path).stem
            identity = name.rsplit('_aug', 1)[0] if '_aug' in name else name
            identity_files[identity].append(file_path)
        
        triplets, identities = [], list(identity_files.keys())
        for _ in range(50000):
            anchor_id = random.choice(identities)
            if len(identity_files[anchor_id]) < 2:
                continue
            anchor, positive = random.sample(identity_files[anchor_id], 2)
            neg_id = random.choice([i for i in identities if i != anchor_id])
            negative = random.choice(identity_files[neg_id])
            triplets.append((anchor, positive, negative))
        
        with open(self.dirs['metadata'] / 'train_triplets.pkl', 'wb') as f:
            pickle.dump(triplets, f)
        print(f"  生成 {len(triplets)} 个三元组")
        return triplets
    
    def run(self):
        data_info = self.analyze_current_data()
        splits = self.create_retrieval_splits(data_info)
        file_mapping = self.copy_files_for_split(splits, data_info)
        self.create_metadata(splits, file_mapping)
        self.create_train_pairs_for_metric_learning(file_mapping)

        print("\n" + "="*60)
        print("✅ 数据重组织完成！")
        print("="*60)
        print(f"输出目录: {self.output_dir}")
        print(f"├── train/   {len(file_mapping['train'])} 个文件")
        print(f"├── gallery/ {len(file_mapping['gallery']['files'])} 个文件（{len(file_mapping['gallery']['label_map'])} 身份）")
        print(f"├── query/   {len(file_mapping['query']['files'])} 个文件（seen {file_mapping['query']['n_seen']} | unseen {file_mapping['query']['n_unseen']}）")
        print(f"└── metadata/ retrieval_metadata.json, *_files.txt, labels.pkl, train_triplets.pkl")
        return

# 使用示例
if __name__ == "__main__":
    organizer = RetrievalDataOrganizer(
        augmented_dir='/home/jz97/3d_face_repo/augmented_data',
        output_dir='/home/jz97/3d_face_repo/retrieval_data',
        seed=42,
        reset_output=True,  # 清空旧文件，强烈建议开启
    )
    organizer.run()
