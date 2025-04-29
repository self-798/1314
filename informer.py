import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import math
import seaborn as sns
from math import sqrt
from glob import glob
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
# 设置随机种子，确保结果可重复
torch.manual_seed(42)
np.random.seed(42)
import os
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
# 检查CUDA是否可用并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ProbMask for the ProbAttention mechanism
class ProbMask():
    def __init__(self, B, H, L, index, u_part, S, device):
        self.B = B
        self.H = H
        self.L = L
        self.S = S
        self.index = index
        self.device = device
        
        # Initialize mask
        self.mask = torch.ones(B, H, L, S, dtype=torch.bool, device=device)
        for i in range(B):
            for j in range(H):
                self.mask[i, j, :, :] = torch.scatter(
                    self.mask[i, j, :, :],
                    dim=-1,
                    index=index[i, j, :],
                    src=torch.zeros(L, u_part, dtype=torch.bool, device=device)
                )

class ProbAttention(nn.Module):
    """ProbSparse自注意力机制"""
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        
    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L, E = K.shape
        _, _, S, _ = Q.shape
        
        # 计算采样的k值
        K_expand = K.unsqueeze(-3).expand(B, H, S, L, E)
        index_sample = torch.randint(L, (S, sample_k))
        K_sample = K_expand[:, :, torch.arange(S).unsqueeze(1), index_sample, :]
        
        # 计算Q和K_sample的注意力分数
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
        
        # 找到最重要的前n_top个key
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L)
        M_top = M.topk(n_top, sorted=False)[1]
        
        # 使用n_top个最重要的query
        Q_reduce = Q[torch.arange(B)[:, None, None], 
                     torch.arange(H)[None, :, None], 
                     M_top, :]
        
        # 计算完整的Q和K的注意力分数
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        
        return Q_K, M_top
    
    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, D)
        else:
            contex = torch.zeros((B, H, L_Q, D), device=V.device)
            
        return contex
    
    def forward(self, queries, keys, values):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape
        
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        
        U = self.factor * np.ceil(np.log(S)).astype('int').item()
        u = self.factor * np.ceil(np.log(L)).astype('int').item()
        
        # 减少采样的query和key的数量
        U_part = min(U, L)
        u_part = min(u, S)
        
        score_K, index = self._prob_QK(queries, keys, u_part, U_part)
        
        # 获取上下文
        scale = self.scale or 1. / math.sqrt(D)
        
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L, index, u_part, S, device=queries.device)
            scores = torch.masked_fill(score_K * scale, attn_mask.mask, -np.inf)
            attention_weights = torch.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            context = torch.matmul(attention_weights, values)
        else:
            attention_weights = torch.softmax(score_K * scale, dim=-1)
            attention_weights = self.dropout(attention_weights)
            context = torch.matmul(attention_weights, values)
        
        return context, attention_weights

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 正弦和余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为缓冲区而非参数
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]

# Informer模型（分类输出）
class Informer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, n_heads=4, n_layers=2, dropout=0.3, 
                 factor=5, max_seq_len=100, num_classes=3):
        super(Informer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.input_dropout = nn.Dropout(dropout)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        
        # 输出层 - 修改为分类输出
        self.output_dropout = nn.Dropout(dropout)
        # 每个输出维度有num_classes个可能的类别
        self.fc = nn.Linear(d_model, output_dim * num_classes)
        self.num_classes = num_classes
        self.output_dim = output_dim

    def forward(self, x):
        # 如果输入是二维的，添加时间维度
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 输入投影
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        
        # 应用位置编码
        x = self.pos_encoder(x)
        x = self.input_dropout(x)
        
        # 编码器处理
        for encoder in self.encoder_layers:
            x = encoder(x)
        
        # 输出处理
        x = self.output_dropout(x[:, -1, :])  # 取最后一个时间步
        x = self.fc(x)
        
        # 重塑为[batch_size, output_dim, num_classes]并应用softmax
        x = x.reshape(-1, self.output_dim, self.num_classes)
        
        return x

# 改进的数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        """
        X: 形状为 [sequences, seq_length, features] 的序列特征数据
        y: 形状为 [sequences, targets] 的目标数据
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # 使用long类型用于分类

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 新的数据加载与处理函数
def load_data_from_directory(base_path):
    """
    从指定目录加载所有csv文件并组织成时间序列数据
    """
    # 获取目录下的所有CSV文件
    # Get all CSV files in the directory
    all_files = glob(os.path.join(data_dir, "snapshot_sym*.csv"))
    
    # Extract dates from filenames
#     files = [
#     f"snapshot_sym{j}_date{i}_{session}.csv"
#     for j in range(0, 5)  # sym_1 到 sym_5
#     for i in range(0,10)  # date_0 到 date_7
#     for session in ['am', 'pm']
# ]
#     file_path = os.path.join(data_dir, files)
    if not all_files:
        raise ValueError(f"在目录 {data_dir} 中未找到数据文件")
    
    print(f"找到 {len(all_files)} 个CSV文件")
    
    # 加载所有数据
    all_dfs = []
    for file in all_files:
        file_path = os.path.join(base_path, file)
        if os.path.exists(file_path):
            temp_df = pd.read_csv(file_path)
            all_dfs.append(temp_df)
            print(f"加载文件: {file_path}，数据形状: {temp_df.shape}")
        else:
            print(f"文件 {file} 不存在，跳过。")
    
    # 合并所有数据
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"合并后数据形状: {combined_df.shape}")
    
    # 确保时间列是字符串格式
    combined_df['time'] = combined_df['time'].astype(str)
    combined_df['time'] = combined_df['time'].str.zfill(8)  # 补齐时间格式，如'9:30:00'变为'09:30:00'
    
    # 对amount_delta进行标准化处理（如果它未被标准化）
    if 'amount_delta' in combined_df.columns:
        # 按股票和日期分组计算z-score
        def normalize_amount_delta(group):
            if len(group) > 1:
                # 修复clip函数的参数问题，使用适用于所有版本的写法
                std = group['amount_delta'].std()
                if std == 0:
                    std = 1.0  # 防止除以0
                group['amount_delta'] = (group['amount_delta'] - group['amount_delta'].mean()) / std
            return group
        
        combined_df = combined_df.groupby(['sym', 'date']).apply(normalize_amount_delta).reset_index(drop=True)
    
    return combined_df

def forward_looking_normalization(df, feature_cols, window_size=50):
    """
    使用前向滑动窗口进行标准化，确保t时刻的标准化只使用t之前的数据
    """
    df = df.copy()
    normalized_df = df.copy()
    
    # # 按股票和日期分组
    # for (sym, date), group in df.groupby(['sym', 'date']):
    #     group = group.sort_values('time')
    #     group_idx = group.index
        
    #     for col in feature_cols:
    #         # 跳过已标准化的列（以'n_'开头的列）
    #         if col.startswith('n_') and col != 'n_midprice':
    #             continue
                
    #         # 初始化存储标准化后的值
    #         normalized_values = np.zeros(len(group))
            
    #         for i in range(len(group)):
    #             if i < window_size:
    #                 # 对前window_size个样本，使用前i个样本的统计量
    #                 if i == 0:
    #                     # 第一个样本无法标准化，设为0
    #                     normalized_values[i] = 0
    #                 else:
    #                     window_mean = group[col].iloc[:i].mean()
    #                     window_std = group[col].iloc[:i].std()
    #                     if window_std == 0:
    #                         window_std = 1
    #                     normalized_values[i] = (group[col].iloc[i] - window_mean) / window_std
    #             else:
    #                 # 使用过去window_size个样本计算均值和标准差
    #                 window_mean = group[col].iloc[i-window_size:i].mean()
    #                 window_std = group[col].iloc[i-window_size:i].std()
    #                 if window_std == 0:
    #                     window_std = 1
    #                 normalized_values[i] = (group[col].iloc[i] - window_mean) / window_std
            
    #         # 将标准化后的值分配回DataFrame
    #         normalized_df.loc[group_idx, col] = normalized_values
    
    return normalized_df

def time_based_split(df, test_ratio=0.2):
    """
    按时间顺序分割数据集，前(1-test_ratio)的日期为训练集，后test_ratio的日期为测试集
    """
    # 获取日期的唯一值并排序
    unique_dates = sorted(df['date'].unique())
    
    # 确定分割点
    split_idx = int(len(unique_dates) * (1 - test_ratio))
    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]
    
    print(f"训练集日期: {train_dates}")
    print(f"测试集日期: {test_dates}")
    
    # 分割数据
    train_df = df[df['date'].isin(train_dates)]
    test_df = df[df['date'].isin(test_dates)]
    
    return train_df, test_df

def create_time_series_sequences(df, seq_length, feature_cols, target_cols):
    """
    使用Pandas和NumPy高效创建时间序列窗口的替代方法
    """
    X_list = []
    y_list = []
    
    # 对于每个(股票,日期)组，一次性创建所有窗口
    for (sym, date), group in tqdm(df.groupby(['sym', 'date']), desc="创建时间序列窗口"):
        group = group.sort_values('time').reset_index(drop=True)
        
        if len(group) >= seq_length + 1:
            # 提取特征矩阵和目标矩阵
            X_matrix = group[feature_cols].values
            y_matrix = group[target_cols].values
            
            # 创建索引数组，用于切片
            indices = np.arange(len(group) - seq_length)
            
            # 使用向量化操作同时创建所有窗口
            windows = np.array([X_matrix[i:i+seq_length] for i in indices])
            targets = y_matrix[indices + seq_length]
            
            X_list.append(windows)
            y_list.append(targets)
    
    # 如果至少有一个有效序列
    if X_list:
        X_sequences = np.vstack(X_list)
        y_sequences = np.vstack(y_list)
        return X_sequences, y_sequences
    else:
        return np.array([]), np.array([])

def convert_to_class_labels(y):
    """将连续值转换为类别标签: 0=下降, 1=稳定, 2=上升"""
    y_classes = np.zeros_like(y, dtype=np.int64)
    
    # 下降 (<0.73)
    y_classes[y < 0.73] = 0
    
    # 稳定 (0.73-1.2)
    y_classes[(y >= 0.73) & (y <= 1.2)] = 1
    
    # 上升 (>1.2)
    y_classes[y > 1.2] = 2
    
    return y_classes

def balance_classes(X, y):
    """
    通过过采样少数类和欠采样多数类来平衡各个类别
    """
    # 统计各类别数量
    unique_classes = np.unique(y.reshape(-1))  # 展平y来计算所有类别
    class_counts = {}
    
    # 统计每个目标维度每个类别的数量
    for i in range(y.shape[1]):
        for cls in unique_classes:
            if (i, cls) not in class_counts:
                class_counts[(i, cls)] = 0
            class_counts[(i, cls)] += np.sum(y[:, i] == cls)
    
    # 计算每个维度的目标样本数（使用中位数作为平衡点）
    target_counts = {}
    for i in range(y.shape[1]):
        counts = [class_counts.get((i, cls), 0) for cls in unique_classes]
        target_counts[i] = int(np.median(counts))  # 使用中位数作为平衡点
    
    # 创建空数组，准备存放平衡后的数据
    indices_to_keep = []
    
    # 对每个类别进行过采样或欠采样
    for i in range(y.shape[1]):
        # 为每个维度分别处理
        for cls in unique_classes:
            current_count = class_counts.get((i, cls), 0)
            # 找出该类别对应的索引
            indices = np.where(y[:, i] == cls)[0]
            
            if len(indices) == 0:
                continue
                
            if current_count > target_counts[i]:
                # 欠采样（随机选择目标数量的样本）
                selected_indices = np.random.choice(indices, size=target_counts[i], replace=False)
                indices_to_keep.extend(selected_indices)
            else:
                # 过采样（使用重复采样）
                # 先添加所有原始样本
                indices_to_keep.extend(indices)
                
                # 然后随机重复采样，直到达到目标数量
                if current_count < target_counts[i]:
                    additional_needed = target_counts[i] - current_count
                    # 使用放回抽样实现过采样
                    additional_indices = np.random.choice(indices, size=additional_needed, replace=True)
                    indices_to_keep.extend(additional_indices)
    
    # 去除重复的索引
    indices_to_keep = list(set(indices_to_keep))
    
    # 提取平衡后的数据集
    X_balanced = X[indices_to_keep]
    y_balanced = y[indices_to_keep]
    
    print(f"原始数据集大小: {X.shape[0]}, 平衡后数据集大小: {X_balanced.shape[0]}")
    return X_balanced, y_balanced

# 完整的数据加载与预处理函数
def load_data(data_dir, seq_length=10, test_ratio=0.2, window_size=50, balance=True, batch_size = 64):
    """
    加载并处理时间序列数据
    
    参数:
    - data_dir: 数据目录
    - seq_length: 序列长度
    - test_ratio: 测试集比例
    - window_size: 标准化窗口大小
    - balance: 是否平衡类别
    
    返回:
    - train_loader: 训练数据加载器
    - test_loader: 测试数据加载器
    - input_dim: 输入特征维度
    - output_dim: 输出标签维度
    """
    # 1. 加载数据
    df = load_data_from_directory(data_dir)
    
    # 2. 确定特征列和标签列
    feature_cols = [col for col in df.columns if col not in ['date', 'time', 'sym', 'label_5', 'label_10', 'label_20', 'label_40', 'label_60']]
    target_cols = ['label_5', 'label_10']  # 只使用5分钟和10分钟的标签
    
    # 3. 对需要的特征进行标准化
    df = forward_looking_normalization(df, feature_cols, window_size)
    
    # 4. 移除NaN值
    df = df.dropna()
    
    # 5. 按时间顺序分割数据
    train_df, test_df = time_based_split(df, test_ratio)
    
    # 6. 创建序列
    X_train, y_train = create_time_series_sequences(train_df, seq_length, feature_cols, target_cols)
    X_test, y_test = create_time_series_sequences(test_df, seq_length, feature_cols, target_cols)
    
    # 7. 将标签转换为类别
    y_train_classes = convert_to_class_labels(y_train)
    y_test_classes = convert_to_class_labels(y_test)
    
    # 8. 平衡训练集类别
    if balance:
        X_train, y_train_classes = balance_classes(X_train, y_train_classes)
    
    # 9. 显示类别分布
    print("\n训练集类别分布:")
    for i, label in enumerate(target_cols):
        unique, counts = np.unique(y_train_classes[:, i], return_counts=True)
        print(f"{label}: {dict(zip(unique, counts))}")
    
    print("\n测试集类别分布:")
    for i, label in enumerate(target_cols):
        unique, counts = np.unique(y_test_classes[:, i], return_counts=True)
        print(f"{label}: {dict(zip(unique, counts))}")
    
    # 10. 创建数据加载器
    train_dataset = TimeSeriesDataset(X_train, y_train_classes)
    test_dataset = TimeSeriesDataset(X_test, y_test_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 11. 返回模型需要的信息
    input_dim = X_train.shape[2]  # 输入特征维度
    output_dim = y_train.shape[1]  # 输出标签维度
    
    return train_loader, test_loader, input_dim, output_dim

# 添加特殊初始化函数以平衡类别偏好
def initialize_output_layer(model):
    """初始化输出层，使其对不同类别的预测概率更加均衡"""
    if hasattr(model, 'fc'):
        # 对FC层权重进行正常初始化
        nn.init.xavier_uniform_(model.fc.weight)
        
        # 特殊处理偏置项，使其对各类别有均衡的初始偏好
        if model.fc.bias is not None:
            with torch.no_grad():
                # 获取参数
                output_dim = model.output_dim
                num_classes = model.num_classes
                
                # 创建特殊偏置初始值
                bias_shape = model.fc.bias.shape[0]
                bias_init = torch.zeros(bias_shape)
                
                # 对每个输出维度设置略微不同的初始偏好
                for i in range(output_dim):
                    # 下降类(0)给予较高初始偏好
                    bias_init[i*num_classes] = 0.8
                    # 稳定类(1)给予中等初始偏好
                    bias_init[i*num_classes+1] = 0.5
                    # 上升类(2)给予较高初始偏好
                    bias_init[i*num_classes+2] = 0.8
                
                # 应用初始化偏置
                model.fc.bias.copy_(bias_init)
                
                print("已特殊初始化输出层偏置，使模型对各类别有均衡的初始偏好")

# 初始化权重函数
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# 训练函数
def train_model(model, train_loader, criterion, optimizer, epochs=20, device=None, scheduler=None):
    if device is None:
        device = next(model.parameters()).device
    
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        # 在每个epoch开始时添加小扰动
        if epoch > 0 and epoch % 10 == 0:
            print("添加参数扰动以跳出局部最小值...")
            for param in model.parameters():
                noise = torch.randn_like(param) * param.abs().mean() * 0.01
                with torch.no_grad():
                    param.add_(noise)
        for X_batch, y_batch in train_loader:
            # 将数据移至设备
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)  # [batch_size, output_dim, num_classes]
            
            # 计算交叉熵损失
            loss = 0
            for i in range(outputs.size(1)):
                loss += criterion(outputs[:, i], y_batch[:, i])
            l2_reg = 0
            # 添加L2正则化损失
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            loss += 0.001 * l2_reg  # 添加到总损失
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.item()
            
            # 计算准确度
            pred_classes = torch.argmax(outputs, dim=2)
            correct = (pred_classes == y_batch).all(dim=1).sum().item()
            correct_predictions += correct
            total_samples += y_batch.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = (correct_predictions / total_samples) * 100
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%")
        
        # 测试集评估
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                
                test_outputs = model(X_test)
                
                # 计算测试集损失
                batch_test_loss = 0
                for i in range(test_outputs.size(1)):
                    batch_test_loss += criterion(test_outputs[:, i], y_test[:, i])
                test_loss += batch_test_loss.item()
                
                # 计算测试集准确度
                test_pred_classes = torch.argmax(test_outputs, dim=2)
                test_correct += (test_pred_classes == y_test).all(dim=1).sum().item()
                test_total += y_test.size(0)
        
        test_avg_loss = test_loss / len(test_loader)
        test_accuracy = (test_correct / test_total) * 100
        print(f"Test Loss: {test_avg_loss:.6f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # 恢复训练模式
        model.train()
        # 在每个epoch结束时检查梯度范数
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"梯度L2范数: {total_norm:.6f}")
        
        # 学习率调度器步进
        if scheduler is not None:
            scheduler.step()
            
        # 输出当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy (%)')
    plt.savefig('training_curves.png')
    plt.close()
    
    return train_losses, train_accuracies
def weight_init(m):
    if isinstance(m, nn.Linear):
        # 使用更大方差的初始化
        nn.init.xavier_normal_(m.weight, gain=1.5)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.1)  # 小的正偏置
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)

# 主函数
if __name__ == "__main__":
    # 数据目录
    data_dir = r'/mnt/e/OneDrive - CUHK-Shenzhen/FE大二下/良文杯/train_set/train_set'
    epochs = 50
    # 加载数据
    seq_length = 10  # 序列长度
    train_loader, test_loader, input_dim, output_dim = load_data(
        data_dir, seq_length=seq_length, test_ratio=0.2, window_size=50, balance=True, batch_size=1024
    )
    
    print(f"输入特征维度: {input_dim}, 输出维度: {output_dim}")
    
    # 初始化模型
    model = Informer(
        input_dim=input_dim,
        output_dim=output_dim, 
        d_model=256,
        n_heads=16,
        n_layers=3,
        dropout=0.1,
        max_seq_len=seq_length,
        num_classes=3
    )
    model.apply(weight_init)  # 应用自定义初始化
    model = model.to(device)
    # initialize_output_layer(model)
    
    # 使用交叉熵损失
    criterion = nn.CrossEntropyLoss()
    
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=0.002,
        betas=(0.9, 0.999),
        weight_decay=0.01  # 增加正则化
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,               # 每5个epoch重启一次
    T_mult=2,            # 每次重启后周期翻倍
    eta_min=1e-6         # 最小学习率
)
    
    # 训练模型
    train_losses, train_accuracies = train_model(
        model, train_loader, criterion, optimizer, epochs=epochs, device=device, scheduler=scheduler
    )
    
    # 保存模型
    torch.save(model.state_dict(), 'informer_classifier.pth')
    
    # 测试模型
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            
            # 计算损失
            batch_loss = 0
            for i in range(outputs.size(1)):
                batch_loss += criterion(outputs[:, i], y_batch[:, i])
            test_loss += batch_loss.item()
            
            # 获取预测类别
            pred_classes = torch.argmax(outputs, dim=2)
            
            # 收集预测和目标
            all_preds.append(pred_classes.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            
            # 计算准确度
            correct += (pred_classes == y_batch).all(dim=1).sum().item()
            total += y_batch.size(0)
        
        print(f"测试损失: {test_loss/len(test_loader):.6f}")
        print(f"测试准确率: {100.0*correct/total:.2f}%")
    
    # 合并所有批次的预测和目标
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # 可视化与评估
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    class_names = ['Down', 'Stable', 'Up']
    
    # Calculate and display confusion matrix for each output dimension
    for i in range(output_dim):
        label_name = ['5', '10'][i] if i < 2 else str(i)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_targets[:, i], all_preds[:, i], labels=[0, 1, 2])
        
        # Print confusion matrix
        print(f"\n混淆矩阵 - label_{label_name}:")
        print("标签: [0: 下降, 1: 稳定, 2: 上升]")
        print(cm)
        
        # Plot confusion matrix heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f'混淆矩阵 - label_{label_name}', fontsize=16)
        plt.ylabel('真实标签', fontsize=14)
        plt.xlabel('预测标签', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_label_{label_name}.png')
        
        # Calculate and print classification report
        report = classification_report(
            all_targets[:, i], all_preds[:, i],
            labels=[0, 1, 2],
            target_names=class_names
        )
        print(f"\n分类报告 - label_{label_name}:")
        print(report)
        
        # 分析每个类别的性能
        print(f"\n混淆矩阵详细分析 - label_{label_name}:")
        total_samples = cm.sum()
        print(f"总样本数: {total_samples}")
        accuracy = np.trace(cm) / total_samples
        print(f"总体准确率: {accuracy*100:.2f}%")
        
        print("\n各类别指标:")
        for j, class_name in enumerate(class_names):
            class_total = cm[j, :].sum()
            class_correct = cm[j, j]
            class_precision = cm[j, j] / cm[:, j].sum() if cm[:, j].sum() > 0 else 0
            class_recall = class_correct / class_total if class_total > 0 else 0
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
            
            print(f"类别 '{class_name}':")
            print(f"  样本数: {class_total}")
            print(f"  准确预测: {class_correct} ({class_recall*100:.2f}%)")
            print(f"  精确率: {class_precision*100:.2f}%")
            print(f"  召回率: {class_recall*100:.2f}%")
            print(f"  F1分数: {class_f1:.4f}")
        
        print("\n错误分析:")
        for j in range(3):
            for k in range(3):
                if j != k:
                    error_rate = cm[j, k] / cm[j, :].sum() * 100 if cm[j, :].sum() > 0 else 0
                    print(f"  真实'{class_names[j]}'被错误预测为'{class_names[k]}': {cm[j, k]}样本 ({error_rate:.2f}%)")