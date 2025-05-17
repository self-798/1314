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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import os
import math
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
import time
import copy

os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
target_labels = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
seq_length = 100

# 配置参数
train_model = True
train_model_hierarchical = False
multi_label_mode = True  # 是否使用多标签模式
use_focal_loss = False   # 是否使用焦点损失
show_pic = True

# 训练参数
epochs = 30
patience = 10  # 早停耐心值
class_weights = torch.tensor([2.5, 0.38, 2.6], device=device)
batch_size = 1024 * 2  # 批量大小
base_lr = 0.0001
dropout = 0.3

# 数据路径
base_path = r'./train_set/train_set'
files = [
    f"snapshot_sym{j}_date{i}_{session}.csv"
    for j in range(0, 5)  # sym_0 到 sym_4
    for i in range(0, 10)  # date_0 到 date_100
    for session in ['am', 'pm']
]

# 设置图表样式
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
sns.set(style="whitegrid")

# 检查字体是否正常显示中文
import matplotlib.font_manager as fm
fonts = [f.name for f in fm.fontManager.ttflist]
print("Available fonts with Chinese support:", [f for f in fonts if any(x in f for x in ['Hei', 'Song', 'Yuan', 'Kai', 'Quan'])])

# 创建模型保存文件夹
os.makedirs('./models', exist_ok=True)
os.makedirs('./results', exist_ok=True)
os.makedirs('./pict', exist_ok=True)

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

# 多标签时间序列数据集类（GPU优化版）
class MultiLabelTimeSeriesDatasetGPU(Dataset):
    def __init__(self, df, feature_cols, target_labels, seq_length, preload_gpu=True):
        # 预处理特征
        if preload_gpu:
            self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(device)
        else:
            self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        
        # 预处理多个标签
        self.targets = {}
        for label in target_labels:
            if label in df.columns:
                if preload_gpu:
                    self.targets[label] = torch.tensor(df[label].values, dtype=torch.long).to(device)
                else:
                    self.targets[label] = torch.tensor(df[label].values, dtype=torch.long)
        
        self.seq_length = seq_length
        self.preload_gpu = preload_gpu
        self.target_labels = target_labels
        
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
        
        # 获取多个标签的目标值
        targets = {}
        for label in self.target_labels:
            if label in self.targets:
                targets[label] = self.targets[label][actual_idx - 1]  # 最后一个时间步的标签
        
        return features, targets

# 单标签时间序列数据集类（保留原版功能）
class TimeSeriesSequenceDatasetGPU(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_length, dates=None, preload_gpu=True):
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
        target = self.target[actual_idx - 1]  # 最后一个时间步的标签
        
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
        
        # 标准的正弦余弦位置编码
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
    
class FeatureWiseStandardization(nn.Module):
    """
    对每个样本的每个特征单独标准化（保留时间维度变化）
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        # x: [batch_size, seq_len, features]
        
        # 计算每个样本每个特征通道的均值和标准差
        # 只沿着seq_len维度计算
        mean = torch.mean(x, dim=1, keepdim=True)  # [batch_size, 1, features]
        std = torch.std(x, dim=1, keepdim=True) + self.eps
        
        # 标准化样本，保留时间序列内的变化模式
        return (x - mean) / std
    
# 多标签预测Transformer模型
class MultiLabelTransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.2):
        super(MultiLabelTransformerModel, self).__init__()
        
        # 标签列表
        self.target_labels = target_labels
        self.num_classes = 3  # 假设所有标签都是3分类(下跌、稳定、上涨)
        
        self.sample_norm = FeatureWiseStandardization()  # 特征级标准化(推荐)

        # 特征维度映射到模型维度
        self.input_proj = nn.Linear(input_dim, d_model)

         # 添加另一个LayerNorm用于稳定特征分布
        self.input_norm = nn.LayerNorm(d_model)

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
        
        # 共享特征提取器
        self.shared_features = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 为每个标签创建单独的分类头
        self.classifiers = nn.ModuleDict()
        for i, label in enumerate(self.target_labels):
            # 根据预测窗口长度调整网络复杂度
            window_size = int(label.split('_')[1])
            hidden_size = min(dim_feedforward, int(dim_feedforward * (0.5 + window_size/60)))
            
            self.classifiers[label] = nn.Sequential(
                nn.Linear(dim_feedforward, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout * (0.8 + 0.4 * i/len(self.target_labels))),  # 远期预测需要更多正则化
                nn.Linear(hidden_size, hidden_size//2),
                nn.GELU(),
                nn.Dropout(dropout * 0.8),
                nn.Linear(hidden_size//2, self.num_classes)
            )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, target_label=None):
        # x shape: [batch_size, seq_len, input_dim]
        # 1. 应用样本级标准化 - 在投影前
        x = self.sample_norm(x)

        # 投影到d_model维度
        x = self.input_proj(x)

         # 3. 应用LayerNorm稳定特征分布
        x = self.input_norm(x)

        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 通过Transformer编码器
        x = self.transformer_encoder(x)
        
        # 取序列的平均值作为特征表示
        x = torch.mean(x, dim=1)
        
        # 提取共享特征
        shared_features = self.shared_features(x)
        
        # 如果指定了目标标签，只返回该标签的预测
        if target_label is not None and target_label in self.target_labels:
            return self.classifiers[target_label](shared_features)
        
        # 否则返回所有标签的预测
        outputs = {}
        for label in self.target_labels:
            outputs[label] = self.classifiers[label](shared_features)
            
        return outputs

# 单标签Transformer模型（保留原版）
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # 特征维度映射到模型维度
        self.input_proj = nn.Linear(input_dim, d_model)
        
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
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
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
        # 投影到d_model维度
        x = self.input_proj(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 通过Transformer编码器
        x = self.transformer_encoder(x)
        
        # 取序列的平均值作为特征表示
        x = torch.mean(x, dim=1)
        
        # 分类
        x = self.classifier(x)
        return x

# 训练多标签模型函数
def train_multi_label_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                           epochs=20, patience=5, task_weights=None):
    """训练多标签模型"""
    print("\n开始训练多标签模型...")
    
    # 如果没有指定任务权重，则所有任务权重相等
    if task_weights is None:
        task_weights = {label: 1.0 for label in model.target_labels}
    # 创建模型名称和检查点路径
    timestamp = datetime.now().strftime("%m%d_%H%M")
    checkpoint_dir = './models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_name = f"multi_label_transformer_{timestamp}"
    checkpoint_path = f"{checkpoint_dir}/multi_label_transformer_0501_2021_checkpoint.pth"
    best_model_path = f"./models/{model_name}_best.pth"
    
    # Check if there's an existing checkpoint and load it if available
    existing_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('_checkpoint.pth')]
    if existing_checkpoints :
        latest_checkpoint = sorted(existing_checkpoints)[-1]
        checkpoint_path = f"{checkpoint_dir}/{latest_checkpoint}"
        print(f"Found existing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_acc = checkpoint.get('best_val_acc', 0)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 0)} with validation accuracy: {best_val_acc:.2f}%")
    else:
        print("没有找到现有检查点，开始新的训练。")

    print(f"模型名称: {model_name}")
    print(f"最佳模型将保存到: {best_model_path}")
    print(f"训练检查点将保存到: {checkpoint_path}")
    best_val_acc = 0
    no_improve = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # 训练阶段
        model.train()
        total_loss = 0
        correct_samples = {label: 0 for label in model.target_labels}
        total_samples = {label: 0 for label in model.target_labels}
        
        for batch_X, batch_y_dict in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # 计算每个标签的损失并累加
            batch_loss = 0
            for label in model.target_labels:
                if label in batch_y_dict:
                    label_loss = criterion(outputs[label], batch_y_dict[label])
                    batch_loss += task_weights[label] * label_loss
                    
                    # 计算准确率
                    _, predicted = torch.max(outputs[label], 1)
                    total_samples[label] += batch_y_dict[label].size(0)
                    correct_samples[label] += (predicted == batch_y_dict[label]).sum().item()
            
            # 反向传播
            batch_loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += batch_loss.item()
        
        # 计算平均训练损失和准确率
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        avg_train_acc = 0
        for label in model.target_labels:
            if total_samples[label] > 0:
                label_acc = 100 * correct_samples[label] / total_samples[label]
                avg_train_acc += label_acc
        avg_train_acc = avg_train_acc / len(model.target_labels)
        train_accs.append(avg_train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct_samples = {label: 0 for label in model.target_labels}
        total_samples = {label: 0 for label in model.target_labels}
        
        with torch.no_grad():
            for batch_X, batch_y_dict in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                outputs = model(batch_X)
                
                # 计算每个标签的损失并累加
                batch_loss = 0
                for label in model.target_labels:
                    if label in batch_y_dict:
                        label_loss = criterion(outputs[label], batch_y_dict[label])
                        batch_loss += task_weights[label] * label_loss
                        
                        # 计算准确率
                        _, predicted = torch.max(outputs[label], 1)
                        total_samples[label] += batch_y_dict[label].size(0)
                        correct_samples[label] += (predicted == batch_y_dict[label]).sum().item()
                
                val_loss += batch_loss.item()
        
        # 计算平均验证损失和准确率
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        all_accs = []
        for label in model.target_labels:
            if total_samples[label] > 0:
                label_acc = 100 * correct_samples[label] / total_samples[label]
                all_accs.append(label_acc)
                print(f"  - {label} 验证准确率: {label_acc:.2f}%")
            
        avg_val_acc = sum(all_accs) / len(all_accs)
        val_accs.append(avg_val_acc)
        
        # 更新学习率
        scheduler.step(avg_val_acc)
        
        # 打印当前轮次的训练结果
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={avg_train_acc:.2f}%, '
              f'Val Loss={val_loss:.4f}, Val Acc={avg_val_acc:.2f}%, Time={epoch_time:.2f}s')
        
        # 保存最佳模型
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # Save checkpoint for possible resuming
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': avg_val_acc,
                'best_val_acc': best_val_acc
            }, checkpoint_path)
            torch.save(model.state_dict(), best_model_path)
            print(f"找到更好的模型，验证准确率: {avg_val_acc:.2f}%")
            no_improve = 0
        else:
            no_improve += 1
            print(f"验证准确率未改善: {no_improve}/{patience}")
        
        # 早停
        if no_improve >= patience:
            print(f"早停: {patience} 轮无改善")
            break
    
    # 训练结束，加载最佳模型权重
    model.load_state_dict(best_model_wts)
    print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    
    # 返回训练历史和最佳准确率
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }
    
    return model, history, best_val_acc

# 评估多标签模型函数
def evaluate_multi_label_model(model, test_loader, device):
    """评估多标签模型性能"""
    model.eval()
    results = {}
    
    for label in model.target_labels:
        results[label] = {
            'predictions': [],
            'true_labels': [],
        }
    
    with torch.no_grad():
        for batch_X, batch_y_dict in tqdm(test_loader, desc="Evaluating model"):
            outputs = model(batch_X)
            
            for label in model.target_labels:
                if label in batch_y_dict:
                    _, predicted = torch.max(outputs[label], 1)
                    results[label]['predictions'].extend(predicted.cpu().numpy())
                    results[label]['true_labels'].extend(batch_y_dict[label].cpu().numpy())
    
    # 计算每个标签的评估指标
    for label in model.target_labels:
        if results[label]['predictions']:  # 确保有预测结果
            y_true = results[label]['true_labels']
            y_pred = results[label]['predictions']
            
            accuracy = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, digits=4)
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            results[label]['accuracy'] = accuracy
            results[label]['report'] = report
            results[label]['conf_matrix'] = conf_matrix
            
            print(f"\n{label} 评估结果:")
            print(f"准确率: {accuracy:.4f}")
            print("分类报告:")
            print(report)
    
    return results

# 加载和预处理数据
def load_data(files):
    """加载数据函数"""
    print("开始加载数据文件...")
    
    # 合并所有文件
    df_list = []
    for file in files:
        file_path = os.path.join(base_path, file)
        if os.path.exists(file_path):
            temp_df = pd.read_csv(file_path)
            df_list.append(temp_df)
        else:
            print(f"文件 {file} 不存在，跳过。")

    # 合并数据集
    print("合并所有数据文件...")
    df = pd.concat(df_list, ignore_index=True)
    print(f"数据加载完成，总行数: {len(df)}")
    return df

# 预处理数据并将其预加载到GPU - 多标签版本
def preprocess_and_load_to_gpu_multi_label(df, feature_cols, target_labels, seq_length):
    """预处理数据并预加载到GPU - 多标签版本"""
    print("预处理数据并加载到GPU（多标签模式）...")
    start_time = time.time()
    
    # 填充空值
    df = df.ffill().bfill()
    
    # 删除不必要的列
    columns_to_drop = []
    if 'date' in df.columns:
        df = df.sort_values('date')
        columns_to_drop.append('date')
    if 'time' in df.columns:
        columns_to_drop.append('time')
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    # 时间序列分割：保持时间顺序性
    # 使用70%数据作为训练集, 15%作为验证集, 15%作为测试集
    n = len(df)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.85)
    
    # 标准化特征
    # scaler = StandardScaler()
    # df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:val_idx]
    test_df = df.iloc[val_idx:]
    
    print(f"训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}, 测试集大小: {len(test_df)}")
    
    # 创建多标签数据集
    print("加载训练集到GPU...")
    train_dataset = MultiLabelTimeSeriesDatasetGPU(train_df, feature_cols, target_labels, seq_length, preload_gpu=True)
    
    print("加载验证集到GPU...")
    val_dataset = MultiLabelTimeSeriesDatasetGPU(val_df, feature_cols, target_labels, seq_length, preload_gpu=True)
    
    print("加载测试集到GPU...")
    test_dataset = MultiLabelTimeSeriesDatasetGPU(test_df, feature_cols, target_labels, seq_length, preload_gpu=True)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    elapsed_time = time.time() - start_time
    print(f"数据预处理和GPU加载完成，耗时：{elapsed_time:.2f}秒")
    
    return train_loader, val_loader, test_loader, len(feature_cols)

# 预处理数据并将其预加载到GPU - 单标签版本（保留原版）
def preprocess_and_load_to_gpu(df, feature_cols, target_col, seq_length):
    """预处理数据并预加载到GPU"""
    print(f"预处理数据并加载到GPU（单标签模式: {target_col}）...")
    start_time = time.time()
    
    # 填充空值
    df = df.ffill().bfill()
    
    # 删除不必要的列
    columns_to_drop = []
    if 'date' in df.columns:
        df = df.sort_values('date')
        columns_to_drop.append('date')
    if 'time' in df.columns:
        columns_to_drop.append('time')
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    # 时间序列分割：保持时间顺序性
    # 使用70%数据作为训练集, 15%作为验证集, 15%作为测试集
    n = len(df)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.85)
    
    # 标准化特征
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:val_idx]
    test_df = df.iloc[val_idx:]
    
    print(f"训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}, 测试集大小: {len(test_df)}")
    
    # 创建GPU优化的数据集
    print("加载训练集到GPU...")
    train_dataset = TimeSeriesSequenceDatasetGPU(train_df, feature_cols, target_col, seq_length, preload_gpu=True)
    
    print("加载验证集到GPU...")
    val_dataset = TimeSeriesSequenceDatasetGPU(val_df, feature_cols, target_col, seq_length, preload_gpu=True)
    
    print("加载测试集到GPU...")
    test_dataset = TimeSeriesSequenceDatasetGPU(test_df, feature_cols, target_col, seq_length, preload_gpu=True)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    elapsed_time = time.time() - start_time
    print(f"数据预处理和GPU加载完成，耗时：{elapsed_time:.2f}秒")
    
    return train_loader, val_loader, test_loader, len(feature_cols)

def main():
    # 加载数据
    start_time = time.time()
    print("开始加载数据文件...")
    df = load_data(files)
    
    # 提取特征列（除标签和日期外）
    print("提取特征列...")
    feature_cols = [col for col in df.columns if col not in ['date', 'time'] and not col.startswith('label_')]
    print(f"使用特征数：{len(feature_cols)}")
    
    # 模型参数
    d_model = 256          # 增加到256维，提高表示能力
    nhead = 8              # 保持8头，但每头现在有32维
    num_layers = 4         # 增加到4层，提高模型深度
    dim_feedforward = 512  # 增加到512，提高非线性变换能力

    # 根据配置选择单标签或多标签模式
    if multi_label_mode:
        # 准备多标签数据
        train_loader, val_loader, test_loader, input_dim = preprocess_and_load_to_gpu_multi_label(
            df, feature_cols, target_labels, seq_length)
        
        # 初始化多标签模型
        print("初始化多标签Transformer模型...")
        multi_model = MultiLabelTransformerModel(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(device)
        
        # 设置多标签模型路径
        multi_model_path = './models/multi_label_transformer_model.pth'
        
        # 检查是否已有模型
        if os.path.exists(multi_model_path) and train_model:
            print(f"加载已有多标签模型: {multi_model_path}")
            multi_model.load_state_dict(torch.load(multi_model_path))
        
        if train_model:
            print("训练多标签Transformer模型...")
            lr = base_lr*(batch_size/256)**0.5  # 根据标签数量调整学习率
            # 设置优化器和损失函数
            optimizer = optim.AdamW(multi_model.parameters(), lr=lr, weight_decay=0.05)
            
            if use_focal_loss:
                criterion = FocalLoss(weight=class_weights, gamma=2.0)
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                
            # 学习率调度器
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='max',
                factor=0.5,
                patience=3,
                verbose=True
            )
            
            # 设置任务权重 - 根据预测窗口远近调整权重
            task_weights = {
                'label_5': 1.0,
                'label_10': 1.1,  # 增加中期预测权重
                'label_20': 1.2,
                'label_40': 1.1,
                'label_60': 1.0   # 近期和远期预测同等重要
            }
            
            # 训练多标签模型
            multi_model, history, best_val_acc = train_multi_label_model(
                multi_model, 
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                device,
                epochs=epochs,
                patience=patience,
                task_weights=task_weights
            )
            
            # 保存训练好的模型
            torch.save(multi_model.state_dict(), multi_model_path)
            print(f"多标签模型已保存: {multi_model_path}")
            
            # 绘制训练历史
            if show_pic:
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(history['train_loss'], label='Training Loss')
                plt.plot(history['val_loss'], label='Validation Loss')
                plt.title('损失曲线')
                plt.xlabel('轮次')
                plt.ylabel('损失')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(history['train_acc'], label='Training Accuracy')
                plt.plot(history['val_acc'], label='Validation Accuracy')
                plt.title('准确率曲线')
                plt.xlabel('轮次')
                plt.ylabel('准确率 (%)')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig('./pict/multi_label_training_history.png')
                plt.close()
        
        # 评估多标签模型
        print("\n评估多标签模型...")
        eval_results = evaluate_multi_label_model(multi_model, test_loader, device)
        
        # 保存评估结果
        results_dir = './results'
        os.makedirs(results_dir, exist_ok=True)
        
        with open(f'{results_dir}/multi_label_model_results.txt', 'w') as f:
            f.write("多标签Transformer模型评估结果\n")
            f.write("============================\n\n")
            
            # 计算平均准确率
            avg_acc = 0
            for label in target_labels:
                if 'accuracy' in eval_results[label]:
                    avg_acc += eval_results[label]['accuracy']
            avg_acc /= len(target_labels)
            
            f.write(f"平均测试准确率: {avg_acc:.4f}\n\n")
            
            for label in target_labels:
                if 'accuracy' in eval_results[label]:
                    f.write(f"--- {label} 结果 ---\n")
                    f.write(f"准确率: {eval_results[label]['accuracy']:.4f}\n\n")
                    f.write("分类报告:\n")
                    f.write(eval_results[label]['report'])
                    f.write("\n\n混淆矩阵:\n")
                    f.write(str(eval_results[label]['conf_matrix']))
                    f.write("\n\n")
        
        print(f"多标签模型评估结果已保存至: {results_dir}/multi_label_model_results.txt")
        
        # 为每个标签绘制混淆矩阵
        if show_pic:
            for label in target_labels:
                if 'conf_matrix' in eval_results[label]:
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(
                        eval_results[label]['conf_matrix'], 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        xticklabels=['下跌', '稳定', '上涨'],
                        yticklabels=['下跌', '稳定', '上涨']
                    )
                    plt.title(f'{label} 混淆矩阵')
                    plt.ylabel('真实标签')
                    plt.xlabel('预测标签')
                    plt.tight_layout()
                    plt.savefig(f'./pict/{label}_confusion_matrix.png')
                    plt.close()
    
    else:
        # 单标签模式 - 保留原版功能
        for target_col in target_labels:
            print(f"\n处理标签: {target_col}")
            
            # 准备数据
            train_loader, val_loader, test_loader, input_dim = preprocess_and_load_to_gpu(
                df, feature_cols, target_col, seq_length)
            num_classes = 3  
            # 初始化单标签模型
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
            
            # 模型路径
            model_path = f'./models/standard_model_{target_col}.pth'
            
            # 检查是否已有模型
            if os.path.exists(model_path) and not train_model:
                print(f"加载已有标准模型: {model_path}")
                standard_model.load_state_dict(torch.load(model_path))
            
            if train_model:
                print("训练标准Transformer模型...")
                
                # 设置优化器和损失函数
                optimizer = optim.AdamW(standard_model.parameters(), lr=base_lr, weight_decay=0.05)
                
                if use_focal_loss:
                    criterion = FocalLoss(weight=class_weights, gamma=2.0)
                else:
                    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
                
                # 学习率调度器
                scheduler = ReduceLROnPlateau(
                    optimizer, 
                    mode='max',
                    factor=0.5,
                    patience=3,
                    verbose=True
                )
                
                # 训练标准模型
                no_improve_epochs = 0   
                best_val_acc = 0
                train_losses = []
                val_losses = []
                train_accs = []
                val_accs = []
                
                for epoch in range(epochs):
                    epoch_start = time.time()
                    standard_model.train()
                    total_loss = 0
                    correct = 0
                    total = 0
                    
                    # 训练循环
                    for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
                        optimizer.zero_grad()
                        outputs = standard_model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(standard_model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        total_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                    
                    train_loss = total_loss / len(train_loader)
                    train_acc = 100 * correct / total
                    train_losses.append(train_loss)
                    train_accs.append(train_acc)
                    
                    # 验证循环
                    standard_model.eval()
                    val_loss = 0
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for batch_X, batch_y in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                            outputs = standard_model(batch_X)
                            loss = criterion(outputs, batch_y)
                            
                            val_loss += loss.item()
                            _, predicted = torch.max(outputs, 1)
                            val_total += batch_y.size(0)
                            val_correct += (predicted == batch_y).sum().item()
                    
                    val_loss = val_loss / len(val_loader)
                    val_acc = 100 * val_correct / val_total
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)
                    
                    # 更新学习率
                    scheduler.step(val_acc)
                    
                    epoch_time = time.time() - epoch_start
                    print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, '
                          f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, '
                          f'Time={epoch_time:.2f}s')
                    
                    # 保存最佳模型
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(standard_model.state_dict(), model_path)
                        print(f"保存最佳标准模型，验证准确率: {val_acc:.2f}%")
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
                    
                    # 早停
                    if no_improve_epochs >= patience:
                        print(f"早停: {patience} 轮无改善")
                        break
                
                # 绘制训练历史
                if show_pic:
                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    plt.plot(train_losses, label='Training Loss')
                    plt.plot(val_losses, label='Validation Loss')
                    plt.title('损失曲线')
                    plt.xlabel('轮次')
                    plt.ylabel('损失')
                    plt.legend()
                    
                    plt.subplot(1, 2, 2)
                    plt.plot(train_accs, label='Training Accuracy')
                    plt.plot(val_accs, label='Validation Accuracy')
                    plt.title('准确率曲线')
                    plt.xlabel('轮次')
                    plt.ylabel('准确率 (%)')
                    plt.legend()
                    
                    plt.tight_layout()
                    pict_dir = f'./pict/{target_col}'
                    os.makedirs(pict_dir, exist_ok=True)
                    plt.savefig(f'{pict_dir}/training_history.png')
                    plt.close()
            
            # 评估单标签模型
            print("\n评估标准模型...")
            standard_model.load_state_dict(torch.load(model_path))
            standard_model.eval()
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in tqdm(test_loader, desc="Evaluating standard model"):
                    outputs = standard_model(batch_X)
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
            
            accuracy = accuracy_score(all_labels, all_preds)
            report = classification_report(all_labels, all_preds, digits=4)
            conf_matrix = confusion_matrix(all_labels, all_preds)
            
            print(f"\n标准Transformer模型评估 ({target_col}):")
            print(f"测试准确率: {accuracy:.4f}")
            print("\n分类报告:")
            print(report)
            
            # 保存评估结果
            results_dir = './results'
            os.makedirs(results_dir, exist_ok=True)
            
            with open(f'{results_dir}/standard_model_results_{target_col}.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(str(report))
                f.write("\n\nConfusion Matrix:\n")
                f.write(str(conf_matrix))
            
            # 绘制混淆矩阵
            if show_pic:
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    conf_matrix, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=['下跌', '稳定', '上涨'],
                    yticklabels=['下跌', '稳定', '上涨']
                )
                plt.title(f'标准模型混淆矩阵 ({target_col})')
                plt.ylabel('真实标签')
                plt.xlabel('预测标签')
                pict_dir = f'./pict/{target_col}'
                os.makedirs(pict_dir, exist_ok=True)
                plt.savefig(f'{pict_dir}/confusion_matrix.png')
                plt.close()
    
    # 总运行时间
    elapsed_time = time.time() - start_time
    print(f"\n总运行时间: {elapsed_time:.2f}秒")
    
    # 打印GPU内存使用情况
    if torch.cuda.is_available():
        print("\nGPU内存使用情况:")
        print(f"已分配内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"缓存内存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

if __name__ == "__main__":
    main()