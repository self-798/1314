
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F  # 添加这一行导入
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
import os
import math
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
import time
import json
class TimeSeriesSequenceDatasetGPU(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_length, dates=None, preload_gpu=True, normalize_seq=False ):
        # 预处理特征和标签，直接转为GPU张量
        if preload_gpu:
            # 直接将整个特征矩阵移至GPU
            self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(device)
            self.target = torch.tensor(df[target_col].values, dtype=torch.long).to(device)
        else:
            # 保留在CPU上，稍后按需加载到GPU
            self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
            self.target = torch.tensor(df[target_col].values, dtype=torch.long)
        
        self.seq_length = seq_length
        self.preload_gpu = preload_gpu
        self.normalize_seq = normalize_seq  # 是否对每个序列进行标准化
        self.feature_cols = feature_cols  # 存储特征列名，用于调试
        
        # 有效索引从序列长度开始
        self.valid_indices = list(range(seq_length, len(df)))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # 获取实际索引
        actual_idx = self.valid_indices[idx]
        start_idx = actual_idx - self.seq_length
        
        # 直接从预加载的张量中切片
        features = self.features[start_idx:actual_idx]
        target = self.target[actual_idx - 1]  
        return features.float(), target.long()
class TimeSeriesPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, is_learnable=True):
        super(TimeSeriesPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.is_learnable = is_learnable
        
        # 初始化位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # 标准的正弦余弦位置编码，但后面会进行改进
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 加入学习型位置编码（可选）
        if is_learnable:
            self.pe = nn.Parameter(pe.unsqueeze(0), requires_grad=True)
        else:
            self.register_buffer('pe', pe.unsqueeze(0))
            
        # 额外添加的时间感知层，用于增强时序特征
        self.time_aware_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        
        # 加入标准位置编码（可学习或固定）
        x = x + self.pe[:, :x.size(1), :]
        
        # 应用时间感知投影，增强时间序列特征
        x = self.time_aware_proj(x)
        
        return self.dropout(x)
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # 特征维度映射到模型维度
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 添加输入标准化层
        self.input_norm = nn.LayerNorm(d_model)
        
        # 全局上下文处理机制
        self.global_context = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        # 使用改进的时间序列位置编码
        self.pos_encoder = TimeSeriesPositionalEncoding(d_model, dropout)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 多级分类头处理不平衡问题
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # 投影到d_model维度
        x = self.input_proj(x)
        
        # 应用输入标准化
        x = self.input_norm(x)
        
        # 计算并融合全局上下文
        global_feat = torch.mean(x, dim=1, keepdim=True)  # [batch, 1, d_model]
        global_feat = self.global_context(global_feat)  # [batch, 1, d_model]
        global_feat = global_feat.expand(-1, x.size(1), -1)  # [batch, seq_len, d_model]
        
        # 残差连接方式融合全局和局部信息
        x = x + global_feat * 0.1  # 使用小系数避免过度影响原始特征
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 通过Transformer编码器
        encoded_features = self.transformer_encoder(x)
        
        # 多种特征聚合方式，捕获不同的时间模式
        # 1. 序列平均值
        mean_pooling = torch.mean(encoded_features, dim=1)
        
        # 2. 最后时间步的特征(最近信息)
        last_step = encoded_features[:, -1, :]
        
        # 结合两种特征表示
        x = mean_pooling + last_step * 0.5
        
        # 分类
        x = self.classifier(x)
        return x
def preprocess_and_load_to_gpu(df, feature_cols, target_col, seq_length):
    """预处理数据并预加载到GPU"""
    print("预处理数据并加载到GPU...")
    start_time = time.time()
    
    # 使用非弃用的方法填充空值
    df = df.ffill().bfill()
    
    # 检查并删除列，仅当它们存在时
    columns_to_drop = []
    if 'date' in df.columns:
        # 按时间顺序排序数据（如果数据不是已排序的）
        df = df.sort_values('date')
        columns_to_drop.append('date')
    
    # 如果有要删除的列，则删除它们
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    # 时间序列分割：保持时间顺序性
    # 使用70%数据作为训练集, 15%作为验证集, 15%作为测试集
    n = len(df)
    train_idx = int(n * 0.4)
    val_idx = int(n * 0.6)
    
    # 标准化特征 - 只使用训练集数据计算统计量，避免数据泄露
    scaler = StandardScaler()
    train_features = df.iloc[:train_idx][feature_cols]
    # 计算自适应类权重
    labels = df.iloc[:train_idx][target_col].values
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    print(f"Class distribution in training data: {dict(zip(unique_classes, class_counts))}")

    # 计算反比权重
    total_samples = len(labels)
    class_weights_np = total_samples / (len(unique_classes) * class_counts)

    # 归一化权重使其平均值为1
    class_weights_np = class_weights_np / class_weights_np.mean()
    print(f"Calculated adaptive class weights: {class_weights_np}")

    # 转换为PyTorch张量并移至设备
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    # 先用训练集的数据拟合scaler
    scaler.fit(train_features)
    # 保存标准化器的均值和标准差
    scaler_params_path = './scaler_params.json'
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        # 换行以提高可读性
        'scale': scaler.scale_.tolist(),
        'columns': feature_cols  # 保存列名
    }
    with open(scaler_params_path, 'w') as f:
        json.dump(scaler_params, f, indent=4)  # 使用缩进格式化JSON
    print(f"标准化器参数和列名已保存到 {scaler_params_path}")
    scaler.save_params = scaler.mean_, scaler.scale_
    
    df[feature_cols] = scaler.transform(df[feature_cols])
    
    test_df = df.iloc[val_idx:]
    
    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:val_idx]

        
    print(f"训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}, 测试集大小: {len(test_df)}")
    # Calculate and print the number of days in each split for monitoring purposes
    # 创建GPU优化的数据集
    print("加载训练集到GPU...")
    train_dataset = TimeSeriesSequenceDatasetGPU(train_df, feature_cols, target_col, seq_length, preload_gpu=True)
    
    print("加载验证集到GPU...")
    val_dataset = TimeSeriesSequenceDatasetGPU(val_df, feature_cols, target_col, seq_length, preload_gpu=True)
    
    print("加载测试集到GPU...")
    test_dataset = TimeSeriesSequenceDatasetGPU(test_df, feature_cols, target_col, seq_length, preload_gpu=True)
    
    # 创建数据加载器
    # 注意：由于数据已经在GPU上，我们不需要pin_memory和num_workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    elapsed_time = time.time() - start_time
    print(f"数据预处理和GPU加载完成，耗时：{elapsed_time:.2f}秒")
    
    return train_loader, val_loader, test_loader, len(feature_cols) , class_weights