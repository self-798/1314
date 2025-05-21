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

os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
target_labels = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
seq_length = 100

train_model = False# 是否训练模型
train_model_hierarchical =  False # 是否训练分层模型
show_pic =True # 是否显示图片

target_labels = target_labels[:]
epochs = 50
target_label =['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
patience = 7 # 早停耐心值
batch_size = 1024 * 1 # 增大批量大小以提高GPU利用率
base_lr = 0.0013
dropout = 0.2

base_path = r'./train_set/train_set'
files = [
    f"snapshot_sym{j}_date{i}_{session}.csv"
    for j in range(0, 5)  # sym_0 到 sym_4
    for i in range(0,102)  # date_0 到 date_100
    for session in ['am', 'pm']
]


# 模型参数
d_model = 128  # 模型维度
nhead = 8  # 多头注意力头数
num_layers = 4  # Transformer编码器层数
dim_feedforward = 512  # 前馈网络维度
num_classes = 3  # 类别数：下跌、稳定、上涨


# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建模型保存文件夹
os.makedirs('./models', exist_ok=True)

# 设置图形风格
sns.set(style="whitegrid")

# 焦点损失 - 对难分样本给予更高权重
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# GPU优化版的时间序列数据集类
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
    

# 原始的时间序列数据集类 - 保留为备用
class TimeSeriesSequenceDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_length, dates=None):
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_length = seq_length
        self.target = df[target_col].values
        self.dates = dates
        
        self.valid_indices = np.arange(seq_length, len(df))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        start_idx = idx - self.seq_length
        
        features = self.df.iloc[start_idx:idx][self.feature_cols].values
        target = self.target[idx - 1]  # 最后一个时间步的标签
        
        # 转换为张量
        features = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.long)
        
        return features, target

# 改进的时间序列位置编码
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

# 定义Transformer模型
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


# 预处理数据并将其预加载到GPU
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
    val_idx = int(n * 0)
    
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


def main():
    # 1. 加载数据 - 保持原有代码
    start_time = time.time()
    print("开始加载数据文件...")
    

    # # 合并所有文件
    # df_list = []
    # for file in files:
    #     file_path = os.path.join(base_path, file)
    #     if os.path.exists(file_path):
    #         temp_df = pd.read_csv(file_path)
    #         df_list.append(temp_df)
    #         # print(f"加载文件: {file_path}，数据形状: {temp_df.shape}")
    #     else:
    #         print(f"文件 {file} 不存在，跳过。")

    # # 合并数据集
    # print("合并所有数据文件...")
    # df = pd.concat(df_list, ignore_index=True)
    df = pd.read_pickle('data_return_rates.pkl')
    print(f"数据加载完成，数据形状: {df.shape}")
    # 提取特征列（除标签和日期外）
    print("提取特征列...")
    # Define columns to exclude: metadata, labels, and returns
    exclude_cols = ['date', 'time', 'label_5', 'label_10', 'label_20', 'label_40', 'label_60']
    # Add return columns that will be calculated later
    exclude_cols.extend([f'return_{t}' for t in [5, 10, 20, 40, 60]])
    # Create feature columns list by excluding these columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"特征列: {feature_cols}")
    # 使用优化的数据加载函数
    # Calculate return rates based on mid-price
    print("计算基于中间价格的收益率...")
    returns_df = pd.DataFrame()

    # Calculate returns for each time horizon
    for time_horizon in [5, 10, 20, 40, 60]:
        return_col = f'return_{time_horizon}'
        df[return_col] = (df['n_midprice'].shift(-time_horizon) - df['n_midprice']) 

    # Save the returns to a pickle file
    df.to_pickle('./data_return_rates.pkl')
    print("收益率处理后的数据已保存到 return_rates.pkl")
    print(f"开始数据预处理和GPU加载，使用目标标签: {target_label}")
    train_loader, val_loader, test_loader, input_dim , class_weights = preprocess_and_load_to_gpu(df, feature_cols, target_label, seq_length)
    
    data_prep_time = time.time() - start_time
    print(f"数据准备阶段总耗时: {data_prep_time:.2f}秒")
    
    
    # 初始化标准Transformer模型（用于对比）
    print("初始化标准Transformer模型...")
    standard_model = TransformerModel(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    
    model_path = f'./models/standard_model_{target_label}.pth'
    if os.path.exists(model_path):
        print(f"加载已有标准模型: {model_path}")
        standard_model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        print("没有找到已有模型，开始训练新的模型...")

    # Evaluate standard model on test set
    eval_start = time.time()
    standard_model.eval()
    std_all_preds = []
    std_all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in tqdm(test_loader, desc="Evaluating standard model"):
            # Data already on GPU, no need to transfer
            outputs = standard_model(batch_X)
            _, predicted = torch.max(outputs, 1)
            std_all_preds.extend(predicted.cpu().numpy())
            std_all_labels.extend(batch_y.cpu().numpy())
        
    std_accuracy = accuracy_score(std_all_labels, std_all_preds)
    std_report = classification_report(std_all_labels, std_all_preds, digits=4)
    std_conf_matrix = confusion_matrix(std_all_labels, std_all_preds)
    
    std_eval_time = time.time() - eval_start
    print(f"Standard model evaluation completed, time: {std_eval_time:.2f} seconds")
    
    print("\nStandard Transformer Model Evaluation:")
    print(f"Test Accuracy: {std_accuracy:.4f}")
    print("\nClassification Report:")
    print(std_report)


if __name__ == "__main__":
    for lab in target_labels:
        target_label = lab
        main()