"""
最先进的3D人脸识别模型
使用DGCNN作为骨干网络，ArcFace作为损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from tqdm import tqdm
import math
import json
from collections import defaultdict

# =====================================
# 第1部分：DGCNN (Dynamic Graph CNN)
# 最先进的点云特征提取器
# =====================================

def knn(x, k):
    """
    寻找k近邻
    Args:
        x: (B, F, N) 特征
        k: 近邻数
    Returns:
        idx: (B, N, k) k近邻的索引
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    dist = -xx - inner - xx.transpose(2, 1)      # = -||xi-xj||^2
    idx = dist.topk(k=k+1, dim=-1)[1][:, :, 1:]  # 丢掉自身
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    构建图特征
    Args:
        x: (B, C, N)
        k: 近邻数
        idx: 预计算的近邻索引
    Returns:
        (B, 2C, N, k) 边特征
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    return feature


class EdgeConv(nn.Module):
    """边卷积层"""
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        # 动态 k，防止 k >= N
        k_eff = min(self.k, x.size(-1) - 1)
        x = get_graph_feature(x, k=k_eff)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class DGCNN(nn.Module):
    """
    DGCNN: Dynamic Graph CNN
    Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", ACM TOG 2019
    """
    def __init__(self, output_dim=512, k=20, dropout=0.2):
        super().__init__()
        self.k = k
        
        # EdgeConv层
        self.conv1 = EdgeConv(3, 64, k)
        self.conv2 = EdgeConv(64, 64, k)
        self.conv3 = EdgeConv(64, 128, k)
        self.conv4 = EdgeConv(128, 256, k)
        
        # 聚合层
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # 全局特征
        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        
        self.linear3 = nn.Linear(256, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, 3) 点云
        Returns:
            (B, output_dim) 特征向量
        """
        batch_size = x.size(0)
        
        # 转置以适应conv层
        x = x.transpose(2, 1)  # (B, 3, N)
        
        # EdgeConv层
        x1 = self.conv1(x)  # (B, 64, N)
        x2 = self.conv2(x1)  # (B, 64, N)
        x3 = self.conv3(x2)  # (B, 128, N)
        x4 = self.conv4(x3)  # (B, 256, N)
        
        # 拼接所有层的特征
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)
        
        # 进一步的特征提取
        x = self.conv5(x)  # (B, 1024, N)
        
        # 全局特征
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # (B, 1024)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # (B, 1024)
        x = torch.cat((x1, x2), 1)  # (B, 2048)
        
        # MLP
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.bn3(self.linear3(x))
        
        # L2归一化（重要！）
        x = F.normalize(x, p=2, dim=1)
        
        return x


# =====================================
# 第2部分：ArcFace损失函数
# 最先进的人脸识别损失
# =====================================

class ArcFaceHead(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        """
        Args:
            in_features: 特征维度 (512)
            out_features: 类别数 (722个身份)
            s: 特征缩放因子
            m: angular margin (弧度)
            easy_margin: 是否使用easy margin
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        
        # 权重矩阵 (类中心)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, input, label):
        """
        Args:
            input: (B, in_features) 归一化的特征
            label: (B,) 标签
        Returns:
            output: (B, out_features) logits
        """
        # 归一化权重
        W = F.normalize(self.weight, p=2, dim=1)
    
        # cos(theta) - 加入clamp防止数值问题
        cosine = F.linear(input, W)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)  # 防止超出[-1,1]
        
        # sin(theta) - 也要clamp
        sine_square = 1.0 - torch.pow(cosine, 2)
        sine_square = sine_square.clamp(min=0)  # 防止负数
        sine = torch.sqrt(sine_square)
            
        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # one-hot
        one_hot = torch.zeros(cosine.size()).to(input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # 输出
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output


# =====================================
# 第3部分：完整的模型
# =====================================

class Face3DModel(nn.Module):
    """完整的3D人脸识别模型"""
    def __init__(self, num_classes=722, feature_dim=512, use_arcface=True):
        """
        Args:
            num_classes: 训练集中的身份数
            feature_dim: 特征维度
            use_arcface: 是否使用ArcFace (训练时True，推理时False)
        """
        super().__init__()
        
        # 特征提取器
        self.backbone = DGCNN(output_dim=feature_dim, k=20, dropout=0.2)
        
        # ArcFace头 (只在训练时使用)
        self.use_arcface = use_arcface
        if use_arcface:
            self.arcface = ArcFaceHead(
                in_features=feature_dim,
                out_features=num_classes,
                s=64.0,  # 缩放因子
                m=0.5   # margin (28.6度)
            )
    
    def forward(self, x, labels=None):
        """
        Args:
            x: (B, N, 3) 点云
            labels: (B,) 标签 (训练时需要)
        Returns:
            训练模式: (features, logits)
            推理模式: features
        """
        # 提取特征
        features = self.backbone(x)  # (B, feature_dim)
        
        if self.training and self.use_arcface:
            assert labels is not None, "训练时需要标签"
            logits = self.arcface(features, labels)
            return features, logits
        else:
            # 推理时只返回特征
            return features
    
    def extract_features(self, x):
        """提取特征的便捷方法"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


# =====================================
# 第4部分：数据集
# =====================================

class FaceDataset(Dataset):
    """3D人脸数据集 - 分类版本"""
    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir: retrieval_data目录
            split: 'train' 或 'val'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        # 获取所有文件和标签
        self.samples = []
        self.labels = []
        self.identity_to_label = {}
        
        # 训练集
        if split == 'train':
            train_dir = self.data_dir / 'train'
            files = sorted(train_dir.glob('*.npz'))
            
            # 创建身份到标签的映射
            identities = set()
            for f in files:
                # 提取身份 (去掉_aug后缀)
                stem = f.stem
                if '_aug' in stem:
                    identity = stem.rsplit('_aug', 1)[0]
                else:
                    identity = stem
                identities.add(identity)
            
            # 分配标签
            for idx, identity in enumerate(sorted(identities)):
                self.identity_to_label[identity] = idx
            
            # 添加样本
            for f in files:
                stem = f.stem
                if '_aug' in stem:
                    identity = stem.rsplit('_aug', 1)[0]
                else:
                    identity = stem
                
                self.samples.append(str(f))
                self.labels.append(self.identity_to_label[identity])
        
        print(f"{split}集: {len(self.samples)} 个样本, {len(self.identity_to_label)} 个身份")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        points = data['points'].astype(np.float32)

        # 兜底成 (N,3)
        if points.ndim == 2 and points.shape[0] == 3 and points.shape[1] != 3:
            points = points.T

        # 零均值 + 单位球（带 epsilon）
        c = points.mean(axis=0, keepdims=True)
        points = points - c
        r = np.linalg.norm(points, axis=1).max()
        points = points / (r + 1e-6)

        label = self.labels[idx]
        return torch.from_numpy(points), label


# =====================================
# 第5部分：训练器
# =====================================

class Trainer:
    """训练管理器"""
    def __init__(self, model, train_loader, val_loader=None, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=0.0001
        )
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        self.best_acc = 0
        self.history = defaultdict(list)
    
    def train_epoch(self, epoch):
        self.model.train()
        losses = []
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for points, labels in pbar:
            points = points.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            features, logits = self.model(points, labels)
            loss = self.criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 统计
            losses.append(loss.item())
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return np.mean(losses), correct/total
    
    def validate(self, epoch):
        if self.val_loader is None:
            return None, None
        
        self.model.eval()
        losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for points, labels in tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]'):
                points = points.to(self.device)
                labels = labels.to(self.device)
                
                features, logits = self.model(points, labels)
                loss = self.criterion(logits, labels)
                
                losses.append(loss.item())
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return np.mean(losses), correct/total
    
    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            print(f'\n--- Epoch {epoch}/{num_epochs} ---')
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            if val_loss:
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            
            # 调整学习率
            self.scheduler.step()
            
            # 打印
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            if val_loss:
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # 保存最佳模型
            if val_acc and val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint('best_model.pth', epoch)
                print(f'✓ 保存最佳模型 (acc: {val_acc:.4f})')
    
    def save_checkpoint(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'history': dict(self.history)
        }, path)


# =====================================
# 第6部分：主函数
# =====================================

def main():
    # 配置
    config = {
        'data_dir': 'retrieval_data',
        'batch_size': 32,
        'num_epochs': 50,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("="*60)
    print("SOTA 3D人脸识别模型训练")
    print("架构: DGCNN + ArcFace")
    print("="*60)
    
    # 1. 数据集
    train_dataset = FaceDataset(config['data_dir'], split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # 2. 模型
    num_classes = len(train_dataset.identity_to_label)
    print(f"\n训练类别数: {num_classes}")
    
    model = Face3DModel(
        num_classes=num_classes,
        feature_dim=512,
        use_arcface=True
    )
    
    # 3. 训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        device=config['device']
    )
    
    trainer.train(config['num_epochs'])
    
    # 4. 保存最终模型（用于推理）
    print("\n保存推理模型...")
    inference_model = Face3DModel(
        num_classes=num_classes,
        feature_dim=512,
        use_arcface=False  # 推理时不需要ArcFace
    )
    
    # 只加载backbone权重
    inference_model.backbone.load_state_dict(model.backbone.state_dict())
    
    torch.save({
        'model_state_dict': inference_model.state_dict(),
        'num_classes': num_classes,
        'config': config
    }, 'dgcnn_arcface_final.pth')
    
    print("✓ 训练完成!")
    print("  训练模型: best_model.pth")
    print("  推理模型: dgcnn_arcface_final.pth")


if __name__ == "__main__":
    main()