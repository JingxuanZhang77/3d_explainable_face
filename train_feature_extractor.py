"""
3D人脸特征提取器训练代码
使用预训练的PointNet++进行迁移学习
目标：学习能够区分不同人脸的特征表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import json
from collections import defaultdict
import random
import logging
from datetime import datetime

# =====================================
# 第1部分：PointNet++模型定义
# 这是核心的3D点云特征提取器
# =====================================

class PointNetSetAbstraction(nn.Module):
    """PointNet++的基础模块：局部特征提取"""
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz, points):
        """
        xyz:    (B, N, 3)
        points: (B, N, C) 或 None
        返回:
        - 中间层: (xyz, per_point_feats) 其中 per_point_feats 为 (B, N, C_out)
        - 最后一层(npoint=None): (xyz, global_feats) 其中 global_feats 为 (B, C_out)
        """
        B, N, _ = xyz.shape

        # 逐点拼接坐标与上层特征，作为输入通道
        if points is not None:
            feats = torch.cat([xyz, points], dim=-1)   # (B, N, 3+C_in)
        else:
            feats = xyz                                 # (B, N, 3)

        # (B, N, C_in) -> (B, C_in, N, 1)，过 1x1 Conv2d 相当于逐点 MLP
        feats = feats.transpose(2, 1).unsqueeze(-1)     # (B, C_in, N, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            feats = F.relu(bn(conv(feats)))             # (B, C_out, N, 1)

        # 还原回逐点特征 (B, N, C_out)
        feats = feats.squeeze(-1).transpose(2, 1)       # (B, N, C_out)

        # 只有最后一层做全局池化 -> (B, C_out)
        if self.npoint is None:
            global_feats = torch.max(feats, dim=1)[0]   # 池化 N 维
            return xyz, global_feats
        else:
            return xyz, feats


class PointNet2Feature(nn.Module):
    """
    预训练的PointNet++特征提取器
    我们会加载在ModelNet40上预训练的权重
    """
    def __init__(self, num_class=40, normal_channel=False):
        super().__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        
        # 特征提取层（这些会从预训练模型加载）
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, in_channel, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024])
        
        # 原始分类头（会被替换）
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)
    
    def forward(self, xyz):
        B, _, _ = xyz.shape
        
        # 逐层提取特征
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # 全局特征
        x = l3_points.view(B, 1024)
        
        # 原始分类路径（训练时会替换）
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        return x


# =====================================
# 第2部分：人脸特征提取器
# 这是我们要训练的模型
# =====================================

class FaceFeatureExtractor(nn.Module):
    """
    人脸特征提取器
    基于预训练的PointNet++，修改为输出固定维度的特征向量
    """
    def __init__(self, pretrained_model=None, feature_dim=512, freeze_layers=2):
        """
        Args:
            pretrained_model: 预训练的PointNet++模型
            feature_dim: 输出特征维度（512维）
            freeze_layers: 冻结前几层（迁移学习策略）
        """
        super().__init__()
        
        # 如果提供了预训练模型，使用它；否则创建新的
        if pretrained_model is not None:
            self.base_model = pretrained_model
            print("✓ 使用预训练模型")
        else:
            self.base_model = PointNet2Feature(num_class=40)
            print("⚠ 创建新模型（无预训练）")
        
        # 冻结底层（迁移学习的关键）
        self._freeze_layers(freeze_layers)
        
        # 替换分类头为特征提取头
        # 这是我们主要训练的部分
        self.feature_head = nn.Sequential(
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(768, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
        # 初始化新层
        self._init_weights()
        
        print(f"模型结构：")
        print(f"  - 基础特征提取器：PointNet++")
        print(f"  - 冻结层数：{freeze_layers}")
        print(f"  - 输出特征维度：{feature_dim}")
    
    def _freeze_layers(self, num_layers):
        """冻结前num_layers层，防止过拟合"""
        layers = [self.base_model.sa1, self.base_model.sa2, self.base_model.sa3]
        
        for i in range(min(num_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False
            print(f"  ✓ 冻结第{i+1}层")
    
    def _init_weights(self):
        """初始化新添加的层"""
        for m in self.feature_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, point_cloud):
        """
        输入：点云 (B, N, 3)
        输出：特征向量 (B, feature_dim)
        """
        B = point_cloud.shape[0]
        
        # 1. 提取基础特征
        # 通过简化的PointNet++提取1024维特征
        with torch.set_grad_enabled(self.training):
            # 逐层提取特征
            l1_xyz, l1_points = self.base_model.sa1(point_cloud, None)
            l2_xyz, l2_points = self.base_model.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.base_model.sa3(l2_xyz, l2_points)
            
            # l3_points已经是 (B, 1024)
            base_features = l3_points
        
        # 2. 投影到人脸特征空间
        features = self.feature_head(base_features)
        
        # 3. L2归一化（重要！让所有特征在单位球面上）
        # 这样相似度就是余弦相似度
        features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def extract_features(self, point_cloud):
        """提取特征的便捷方法（推理时使用）"""
        self.eval()
        with torch.no_grad():
            return self.forward(point_cloud)


# =====================================
# 第3部分：数据集和数据加载
# 处理您的4300个训练样本
# =====================================

class FaceTripletDataset(Dataset):
    """
    三元组数据集
    每个样本返回：(anchor, positive, negative)
    anchor和positive是同一人，negative是不同人
    """
    def __init__(self, data_dir, triplet_file, augment=True):
        """
        Args:
            data_dir: retrieval_data目录
            triplet_file: 预生成的三元组文件
            augment: 是否在线数据增强
        """
        self.data_dir = Path(data_dir)
        
        # 加载预生成的50000个三元组
        with open(triplet_file, 'rb') as f:
            self.triplets = pickle.load(f)
        
        self.augment = augment
        print(f"✓ 加载了 {len(self.triplets)} 个三元组")
        
        # 缓存已加载的点云（避免重复读取）
        self.cache = {}
        self.cache_size = 1000  # 最多缓存1000个
    
    def __len__(self):
        return len(self.triplets)
    
    def _load_point_cloud(self, file_path):
        """加载点云，带缓存机制"""
        if file_path in self.cache:
            return self.cache[file_path].copy()
        
        data = np.load(file_path)
        points = data['points'].astype(np.float32)
        
        # 添加到缓存
        if len(self.cache) < self.cache_size:
            self.cache[file_path] = points
        
        return points
    
    def _augment_point_cloud(self, points):
        """简单的数据增强"""
        if not self.augment or random.random() > 0.5:
            return points
        
        # 随机旋转
        theta = np.random.uniform(0, 2*np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        points = points @ rotation_matrix
        
        # 随机抖动
        noise = np.random.normal(0, 0.001, points.shape).astype(np.float32)
        points = points + noise

        return points.astype(np.float32)
        
    def __getitem__(self, idx):
        """
        返回一个三元组
        """
        anchor_path, positive_path, negative_path = self.triplets[idx]
        
        # 加载点云
        anchor = self._load_point_cloud(anchor_path)
        positive = self._load_point_cloud(positive_path)
        negative = self._load_point_cloud(negative_path)
        
        # 数据增强（可选）
        if self.augment:
            anchor = self._augment_point_cloud(anchor)
            positive = self._augment_point_cloud(positive)
            negative = self._augment_point_cloud(negative)
        
        return (
            torch.from_numpy(anchor.astype(np.float32)),
            torch.from_numpy(positive.astype(np.float32)),
            torch.from_numpy(negative.astype(np.float32)),
        )


# =====================================
# 第4部分：损失函数
# 核心：让同一人的特征靠近，不同人的远离
# =====================================

class TripletMarginLoss(nn.Module):
    """
    三元组损失
    Loss = max(0, d(a,p) - d(a,n) + margin)
    d(a,p): anchor和positive的距离（希望小）
    d(a,n): anchor和negative的距离（希望大）
    margin: 最小间隔
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        计算三元组损失
        所有输入都应该是L2归一化的特征向量
        """
        # 计算欧氏距离
        pos_dist = (anchor - positive).pow(2).sum(dim=1)
        neg_dist = (anchor - negative).pow(2).sum(dim=1)
        
        # 三元组损失
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        # 返回平均损失和一些统计信息
        return {
            'loss': loss.mean(),
            'pos_dist': pos_dist.mean().item(),
            'neg_dist': neg_dist.mean().item(),
            'valid_triplets': (loss > 0).float().mean().item()  # 有效三元组比例
        }


# =====================================
# 第5部分：训练器
# 管理整个训练流程
# =====================================

class Trainer:
    """训练管理器"""
    
    def __init__(self, model, train_loader, val_loader=None, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 损失函数
        self.criterion = TripletMarginLoss(margin=0.5)
        
        # 优化器（不同层使用不同学习率）
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # 训练历史
        self.history = defaultdict(list)
        
        # 最佳模型
        self.best_loss = float('inf')
        
    def _create_optimizer(self):
        """创建优化器，对不同部分使用不同学习率"""
        param_groups = [
            # 预训练层（如果未冻结）：小学习率
            {'params': self.model.base_model.parameters(), 'lr': 1e-5},
            # 新的特征头：大学习率
            {'params': self.model.feature_head.parameters(), 'lr': 1e-3}
        ]
        return torch.optim.Adam(param_groups, weight_decay=1e-4)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = []
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (anchor, positive, negative) in enumerate(pbar):
            # 移到GPU
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            
            # 前向传播：提取特征
            feat_anchor = self.model(anchor)
            feat_positive = self.model(positive)
            feat_negative = self.model(negative)
            
            # 计算损失
            loss_dict = self.criterion(feat_anchor, feat_positive, feat_negative)
            loss = loss_dict['loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录
            epoch_losses.append(loss.item())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pos_d': f'{loss_dict["pos_dist"]:.3f}',
                'neg_d': f'{loss_dict["neg_dist"]:.3f}',
                'valid': f'{loss_dict["valid_triplets"]:.1%}'
            })
        
        return np.mean(epoch_losses)
    
    def validate(self, epoch):
        """验证"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for anchor, positive, negative in tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]'):
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                feat_anchor = self.model(anchor)
                feat_positive = self.model(positive)
                feat_negative = self.model(negative)
                
                loss_dict = self.criterion(feat_anchor, feat_positive, feat_negative)
                val_losses.append(loss_dict['loss'].item())
        
        return np.mean(val_losses)
    
    def train(self, num_epochs, save_dir='checkpoints'):
        """完整训练流程"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("开始训练")
        print("="*60)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")
            
            # 训练
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            # 验证
            val_loss = self.validate(epoch)
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                
                # 学习率调度
                self.scheduler.step(val_loss)
                
                # 保存最佳模型
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(save_dir / 'best_model.pth', epoch)
                    print(f"✓ 保存最佳模型 (val_loss: {val_loss:.4f})")
            
            # 定期保存
            if epoch % 5 == 0:
                self.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch}.pth', epoch)
            
            # 打印总结
            print(f"Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"Val Loss: {val_loss:.4f}")
        
        print("\n✓ 训练完成！")
        return self.history
    
    def save_checkpoint(self, path, epoch):
        """保存检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'history': dict(self.history)
        }, path)


# =====================================
# 第6部分：主函数
# 把所有部分组合起来
# =====================================

def main():
    """主训练流程"""
    
    # 配置
    config = {
        'data_dir': '/home/jz97/3d_face_repo/retrieval_data',
        'batch_size': 32,
        'num_epochs': 15,
        'feature_dim': 512,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'pretrained_path': None  # 如果有预训练模型，在这里指定路径
    }
    
    print("="*60)
    print("3D人脸特征提取器训练")
    print("="*60)
    print(f"配置：")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # 1. 创建数据集
    print("\n加载数据...")
    train_dataset = FaceTripletDataset(
        data_dir=config['data_dir'],
        triplet_file=Path(config['data_dir']) / 'metadata' / 'train_triplets.pkl',
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # 2. 创建模型
    print("\n创建模型...")
    
    # 如果有预训练模型，加载它
    pretrained_model = None
    if config['pretrained_path']:
        print(f"加载预训练模型：{config['pretrained_path']}")
        pretrained_model = PointNet2Feature()
        checkpoint = torch.load(config['pretrained_path'])
        pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    
    model = FaceFeatureExtractor(
        pretrained_model=pretrained_model,
        feature_dim=config['feature_dim'],
        freeze_layers=2  # 冻结前2层
    )
    
    # 3. 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,  # 如果有验证集，在这里添加
        device=config['device']
    )
    
    # 4. 训练
    history = trainer.train(
        num_epochs=config['num_epochs'],
        save_dir='checkpoints'
    )
    
    # 5. 保存最终模型
    print("\n保存最终模型...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, 'face_feature_extractor_final.pth')
    
    print("\n✓ 训练完成！")
    print(f"  最终模型保存在：face_feature_extractor_final.pth")
    print(f"  最佳模型保存在：checkpoints/best_model.pth")
    
    # 6. 绘制训练曲线
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    print(f"  训练曲线保存在：training_history.png")


if __name__ == "__main__":
    main()