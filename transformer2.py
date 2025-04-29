import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR
import os
import math
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_model = False # 是否训练模型
epochs = 20
target_label = 'label_20'
seq_length = 5 
patience = 8 # 早停耐心值
class_weights = torch.tensor([2.5, 0.6, 2.6], device=device)
batch_size = 1024 * 10 # 增大批量大小以提高GPU利用率
base_lr = 0.001
dropout = 0.3
# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
    
# 新的时间序列数据集类（内存高效）
class TimeSeriesSequenceDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_length, dates=None):
        """
        内存高效的时间序列数据集
        
        参数:
            df: 包含所有数据的DataFrame
            feature_cols: 特征列名列表
            target_col: 目标列名
            seq_length: 序列长度
            dates: 可选，要包含的日期列表（用于训练/验证/测试拆分）
        """
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_length = seq_length
        self.indices = []
        
        # 计算有效的索引和对应日期
        print("创建序列索引...")
        groups = self.df.groupby(['sym', 'date'])
        
        for (sym, date), group in tqdm(groups, desc="处理数据组"):
            if dates is not None and date not in dates:
                continue
                
            if len(group) <= seq_length:
                continue
                
            # 获取该组的索引范围
            start_idx = group.index.min()
            end_idx = group.index.max()
            
            # 创建所有有效序列的开始索引
            for i in range(end_idx - start_idx - seq_length + 1):
                self.indices.append((start_idx + i, date))
        
        print(f"创建了 {len(self.indices)} 个序列")
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        start_idx, date = self.indices[idx]
        
        # 一次性获取所有数据以减少DataFrame查询次数
        sequence_data = self.df.iloc[start_idx:start_idx+self.seq_length].reset_index(drop=True)
        
        # 分别提取特征和标签
        X = sequence_data.iloc[:-1][self.feature_cols].values
        y = sequence_data.iloc[-1][self.target_col]
        
        return torch.FloatTensor(X), torch.LongTensor([y])[0]
# 1. 加载数据 - 保持原有代码
base_path = r'./train_set/train_set'
files = [
    f"snapshot_sym{j}_date{i}_{session}.csv"
    for j in range(0, 5)  # sym_0 到 sym_4
    for i in range(0,101)  # date_0 到 date_100
    for session in ['am', 'pm']
]

# 合并所有文件
df_list = []
for file in files:
    file_path = os.path.join(base_path, file)
    if os.path.exists(file_path):
        temp_df = pd.read_csv(file_path)
        df_list.append(temp_df)
        print(f"加载文件: {file_path}，数据形状: {temp_df.shape}")
    else:
        print(f"文件 {file} 不存在，跳过。")

# 合并数据集
df = pd.concat(df_list, ignore_index=True)

# 检查缺失值
print("检查缺失值：")
print(df.isnull().sum())
df = df.dropna()
def create_time_decay_weights(seq_length):
    """创建时间衰减权重，越近的时间点权重越大"""
    weights = torch.linspace(0.5, 1.0, seq_length)
    weights = weights / weights.max()  # 归一化，最大值为1
    return weights


def normalize_new_features(df, new_cols, date_column='date'):
    """只对新添加的特征按日期归一化"""
    result_df = df.copy()
    sys_cols = [col for col in result_df.columns if col.startswith('sys_')]
    new_cols.append("amount_delta") 
    if sys_cols:
        # For system indicator columns (likely one-hot encoded), we should preserve them as is
        for col in sys_cols:
            if col in new_cols:
                new_cols.remove(col)  # Don't normalize one-hot encoded columns
    # 先替换无穷值和NaN
    result_df[new_cols] = result_df[new_cols].replace([np.inf, -np.inf], np.nan)
    result_df[new_cols] = result_df[new_cols].fillna(result_df[new_cols].median())
    
    # 按日期归一化
    means = result_df.groupby(date_column)[new_cols].transform('mean')
    stds = result_df.groupby(date_column)[new_cols].transform('std')
    stds = stds.replace(0, 1)
    
    result_df[new_cols] = (result_df[new_cols] - means) / stds
    return result_df

# 2. 添加技术指标 - 保持原有函数
def add_technical_indicators(df):
    # 确保按时间排序
    df = df.sort_values(['date', 'time'])
    
    # 提取价格相关列
    price_cols = ['n_close', 'n_open', 'n_high', 'n_low', 'n_midprice']
    
    # 按日期分组处理
    grouped = df.groupby('date')
    
    result_dfs = []
    for date, group in grouped:
        # 复制当前日期的数据
        temp_df = group.copy()
        # One-hot encode the 'sym' column if present in the dataframe
        if 'sym' in temp_df.columns:
            # Create one-hot encoding for sym (which should be integers 0-4)
            for i in range(5):  # For sym_0 to sym_4
                temp_df[f'sym_{i}'] = (temp_df['sym'] == i).astype(int)
        # 对每个价格列计算技术指标
        for col in price_cols:
            # 移动平均
            temp_df[f'{col}_ma5'] = temp_df[col].rolling(window=5).mean()
            temp_df[f'{col}_ma10'] = temp_df[col].rolling(window=10).mean()
            if 'sym' in temp_df.columns:
            # Create one-hot encoding for sym (which should be integers 0-4)
                for i in range(5):  # For sym_0 to sym_4
                    temp_df[f'sym_{i}'] = (temp_df['sym'] == i).astype(int)
            # 移动标准差(波动率)
            temp_df[f'{col}_std5'] = temp_df[col].rolling(window=5).std()
            temp_df[f'{col}_std10'] = temp_df[col].rolling(window=10).std()
            
            # 相对强弱指标(RSI)
            delta = temp_df[col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-9)  # 避免除以0
            temp_df[f'{col}_rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = temp_df[col].ewm(span=12).mean()
            ema26 = temp_df[col].ewm(span=26).mean()
            temp_df[f'{col}_macd'] = ema12 - ema26
            temp_df[f'{col}_macd_signal'] = temp_df[f'{col}_macd'].ewm(span=9).mean()
            
            # 动量指标
            temp_df[f'{col}_momentum'] = temp_df[col].pct_change(periods=5)
            
        # 添加额外的交易量指标
        if 'ask_vol1' in temp_df.columns and 'bid_vol1' in temp_df.columns:
            # 买卖压力比
            temp_df['buy_sell_ratio'] = temp_df['bid_vol1'] / temp_df['ask_vol1'].replace(0, 1e-9)
            # 成交量加权平均价格
            if 'n_volume' in temp_df.columns:
                temp_df['vwap'] = (temp_df['n_midprice'] * temp_df['n_volume']).cumsum() / temp_df['n_volume'].cumsum().replace(0, 1e-9)
        
        result_dfs.append(temp_df)
    
    # 合并所有处理后的数据
    result_df = pd.concat(result_dfs)
    
    # 删除生成的NaN值
    result_df = result_df.dropna()
    result_df = result_df.dropna()
    # 仅对新添加的特征进行归一化
    new_cols = [col for col in result_df.columns 
               if col not in df.columns and not col.startswith('label_')]
    
    # 对新特征按日期归一化
    if new_cols:
        result_df = normalize_new_features(result_df, new_cols)
        
    return result_df
# 添加技术指标
df = add_technical_indicators(df)
print("添加因子后数据形状:", df.shape)
print("添加的新特征:", [col for col in df.columns if col not in ['date', 'time','label_5', 'label_10', 'label_20', 'label_40', 'label_60']])

# 检查标签分布
label_counts = df[target_label].value_counts()
print(f"\n标签分布:\n{label_counts}")
print(f"标签比例:\n{df[target_label].value_counts(normalize=True)}")

# 计算每个类别的样本数
class_counts = df[target_label].value_counts()
min_count = min(class_counts)
print(f"最小类别样本数: {min_count}")

# 对数据进行完全平衡处理
def balance_dataset(df, label_col, random_state=42, strategy='undersample_only'):
    balanced_dfs = []
    # 获取每个类别的数据
    class_dfs = {i: df[df[label_col] == i] for i in range(3)}
    
    # 打印原始类别分布
    print("原始类别分布:")
    for i in range(3):
        print(f"类别 {i}: {len(class_dfs[i])} 样本")
    
    # 先计算每个类别的样本数量
    class_counts = [len(df_class) for df_class in class_dfs.values()]
    
    # 决定目标样本数量
    if strategy == 'equal':
        # 所有类别使用相同数量 - 可以调整倍数
        target_samples = int(max(class_counts))
    elif strategy == 'balanced':
        # 所有类别达到中间值
        target_samples = sorted(class_counts)[1]  # 中间类别的数量
    elif strategy == 'undersample_only':
        # 计算中间值的一半
        min_count = min(class_counts)
        # 可以设置为最小类别的1.0-1.2倍，这里用1.0
        target_samples = {
            0: min(len(class_dfs[0]), int(min_count * 1.0)),  # 下跌类
            1: min(len(class_dfs[1]), int(min_count * 1.2)),  # 稳定类
            2: min(len(class_dfs[2]), int(min_count * 1.0))   # 上涨类
        }
    else:
        # 保守策略 - 稍微提升少数类
        min_samples = min(class_counts)
        target_samples = min_samples * 2  # 最小类别的2倍
    
    print(f"目标每类样本数: {target_samples}")
    
    # 对每个类别进行重采样
    for class_label, class_df in class_dfs.items():
        # 获取当前类别的目标样本数
        if isinstance(target_samples, dict):
            target = target_samples[class_label]
        else:
            target = target_samples
            
        if len(class_df) < target:
            # 对少数类进行过采样
            if len(class_df) * 3 < target:
                print(f"警告: 类别 {class_label} 需要大量过采样 ({len(class_df)} → {target})")
            resampled = class_df.sample(n=target, replace=True, random_state=random_state)
        else:
            # 对多数类进行欠采样
            resampled = class_df.sample(n=target, replace=False, random_state=random_state)
        balanced_dfs.append(resampled)
    
    # 合并并打乱平衡后的数据
    balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return balanced_df
# 应用平衡处理
# balanced_df = balance_dataset(df, target_label)

# print(f"平衡后数据形状: {balanced_df.shape}")
# print(f"平衡后标签分布:\n{balanced_df[target_label].value_counts()}")
# print(f"平衡后标签比例:\n{balanced_df[target_label].value_counts(normalize=True)}")

# # 使用平衡后的数据集替换原始数据集
# df = balanced_df

# 3. 按日期进行归一化
def normalize_by_date(df, date_column='date'):
    """
    按日期分组对数据进行标准化
    """
    # 数值型列
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # 排除标签列
    numeric_cols = [col for col in numeric_cols if not col.startswith('label_')]
    # 排除日期列
    numeric_cols.append( [col for col in numeric_cols if col != date_column])
    # 创建结果DataFrame的副本
    result_df = df.copy()
    
    # 将所有数值列转换为float64类型
    result_df[numeric_cols] = result_df[numeric_cols].astype('float64')
    
    # 先替换无穷值为NaN
    result_df[numeric_cols] = result_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # 使用中位数填充NaN值
    result_df[numeric_cols] = result_df[numeric_cols].fillna(result_df[numeric_cols].median())
    
    # 使用transform方法按组计算均值和标准差并直接应用标准化
    print("开始标准化所有日期...")
    
    # 计算每组的均值
    means = result_df.groupby(date_column)[numeric_cols].transform('mean')
    
    # 计算每组的标准差，并处理为0的情况
    stds = result_df.groupby(date_column)[numeric_cols].transform('std')
    stds = stds.replace(0, 1)
    
    # 应用标准化公式
    result_df[numeric_cols] = (result_df[numeric_cols] - means) / stds
    
    # 处理任何剩余的无效值
    result_df[numeric_cols] = result_df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    print("标准化完成!")
    return result_df

# 按日期归一化数据
df_normalized = df
print("归一化后数据形状:", df_normalized.shape)

# 4. 准备序列数据
# 排序以确保时间顺序
df_normalized = df_normalized.sort_values(['sym', 'date', 'time']).reset_index(drop=True)

# 选择特征和标签
feature_cols = [col for col in df_normalized.columns 
               if col not in ['date', 'time' , 'sym'] and not col.startswith('label_')]

# 获取唯一日期并按顺序排列
unique_dates = sorted(df_normalized['date'].unique())
n_dates = len(unique_dates)

# 划分训练、验证和测试集的日期
train_dates = unique_dates[:int(0.8 * n_dates)]
val_dates = unique_dates[int(0.8 * n_dates):int(0.9 * n_dates)]
test_dates = unique_dates[int(0.9 * n_dates):]

print(f"训练集日期: {train_dates[0]} 到 {train_dates[-1]}")
print(f"验证集日期: {val_dates[0]} 到 {val_dates[-1]}")
print(f"测试集日期: {test_dates[0]} 到 {test_dates[-1]}")

# 使用高效数据集类创建数据集
train_dataset = TimeSeriesSequenceDataset(df_normalized, feature_cols, target_label, seq_length, train_dates)
val_dataset = TimeSeriesSequenceDataset(df_normalized, feature_cols, target_label, seq_length, val_dates)
test_dataset = TimeSeriesSequenceDataset(df_normalized, feature_cols, target_label, seq_length, test_dates)

if train_model == False:
    val_dataset = TimeSeriesSequenceDataset(df_normalized, feature_cols, target_label, seq_length, unique_dates)
print(f"训练集: {len(train_dataset)} 序列, 验证集: {len(val_dataset)} 序列, 测试集: {len(test_dataset)} 序列")

# GPU预加载数据集类
class GPUDataset(Dataset):
    def __init__(self, dataset, name="未命名"):
        """将数据集预加载到GPU内存"""
        print(f"正在将{name}数据集加载到GPU内存...")
        
        # 预加载数据(分批处理避免内存溢出)
        features_list = []
        targets_list = []
        
        loader = DataLoader(
            dataset, 
            batch_size=10000,    # 一次性加载整个数据集
            shuffle=False,              
            num_workers=8,              # 增加工作线程
            pin_memory=True,            # 使用固定内存加速传输
            persistent_workers=True     # 保持工作线程活跃状态
        )

        for batch_X, batch_y in tqdm(loader, desc=f"预加载{name}数据"):
            features_list.append(batch_X)
            targets_list.append(batch_y)
        
        # 合并并转移到GPU
        self.features = torch.cat(features_list).to(device)
        self.targets = torch.cat(targets_list).to(device)
        
        # 计算内存使用
        mem_usage = (self.features.element_size() * self.features.nelement() / (1024**3))
        print(f"{name}数据已加载到GPU: {len(self.targets)}样本, {mem_usage:.2f}GB")
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# 初始化数据集变量
train_dataset_gpu = None
val_dataset_gpu = None
use_gpu_train = False
use_gpu_val = False

# 尝试GPU预加载
gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"GPU内存: {gpu_mem:.1f}GB")

if gpu_mem > 8:  # 如果GPU内存超过8GB
    try:
        # 优先预加载较小的验证集
        val_dataset_gpu = GPUDataset(val_dataset, "验证集")
        use_gpu_val = True
        
        # 如果内存足够，也预加载训练集
        if gpu_mem > 16:
            train_dataset_gpu = GPUDataset(train_dataset, "训练集") 
            use_gpu_train = True
    except RuntimeError as e:
        print(f"GPU内存不足: {e}")
        use_gpu_train = False
        use_gpu_val = False

# 创建适当的DataLoader
train_loader = DataLoader(
    train_dataset_gpu if use_gpu_train else train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=0 if use_gpu_train else 4,  # GPU数据不需要工作线程
    pin_memory=not use_gpu_train,  # GPU数据不需要固定内存
    prefetch_factor=2 if not use_gpu_train else None  # GPU数据不需要预取
)

val_loader = DataLoader(
    val_dataset_gpu if use_gpu_val else val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=0 if use_gpu_val else 4,
    pin_memory=not use_gpu_val
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    num_workers=4, 
    pin_memory=True
)

# 5. 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 6. 定义模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 创建因果掩码的Transformer
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.output_linear = nn.Linear(d_model, num_classes)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src shape: [batch_size, seq_length, feature_dim]
        mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        
        # 转换输入维度
        src = self.input_linear(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
        time_weights = create_time_decay_weights(src.size(1)).to(src.device)
        time_weights = time_weights.view(1, -1, 1)  # [1, seq_len, 1]
        src = src * time_weights  # 按时间点加权
        # 应用Transformer编码器（使用因果掩码）- 修改这一行
        output = self.transformer_encoder(src, mask=mask)  # 从src_mask改为mask
        
        # 只使用序列的最后一个时间步进行预测
        output = output[:, -1, :]  # [batch_size, d_model]
        output = self.dropout(output)
        output = self.output_linear(output)
        return output
        
    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码，确保位置i只能关注位置j≤i的信息"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# 7. 初始化模型
input_dim = len(feature_cols)  # 特征维度
d_model = 32   # 模型维度(降低以避免过拟合)
nhead = 2       # 多头注意力中的头数
num_layers = 2  # Transformer编码器层数(降低以避免过拟合)
dim_feedforward = 128 # 前馈网络的维度(降低以避免过拟合)
num_classes = 3  # 类别数：下降(0)、稳定(1)、上升(2)


# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

model = TransformerModel(input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout).to(device)

print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
initialize_weights(model)
# 检查是否有预训练模型
os.makedirs('models', exist_ok=True)
model_path = f'./models/best_seq_transformer_model_{target_label}.pth'
full_model_path = os.path.join(os.path.dirname(__file__), model_path)

# 无论是否训练，都尝试加载已有模型
if os.path.exists(model_path):
    print(f"加载相对路径的模型: {model_path}")
    model.load_state_dict(torch.load(model_path))
elif os.path.exists(full_model_path):
    print(f"加载绝对路径的模型: {full_model_path}")
    model.load_state_dict(torch.load(full_model_path))
else:
    print("没有找到预训练模型，将从头开始训练")
    
if train_model:
    # 8. 训练模型
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    adjusted_lr = base_lr * (batch_size / 256)**0.5
    optimizer = optim.AdamW(
    model.parameters(), 
    lr=adjusted_lr,
    weight_decay=0.05,  # 增加权重衰减
    eps=1e-8,           # 提高数值稳定性
    betas=(0.9, 0.999)  # 标准动量参数
)
    # 添加标签平滑正则化
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑
    # criterion = FocalLoss(weight=class_weights, gamma=2.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',           # 基于验证准确率最大化
        factor=0.5,           # 每次减半
        patience=3,           # 3轮无改善则降低学习率
        verbose=True
    )

    no_improve_epochs = 0   
    best_val_loss = float('inf')
    best_val_acc = 0  # Initialize best validation accuracy
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # No progress bar - simple epoch tracking
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Add gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_acc)  # 使用当前验证准确率
        # 在每个epoch后更新
        # 3. 添加早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Epoch {epoch+1}/{epochs} - 保存新的最佳模型")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"验证准确率已连续{patience}个epoch未改善，停止训练")
            break
            
        print(f"Epoch {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%, 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")

    # 9. 绘制训练过程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.tight_layout()
    plt.savefig('training_history_seq.png')
    plt.show()

# 10. 在测试集上评估最佳模型
model.load_state_dict(torch.load(model_path))
model.eval()

all_preds = []
all_labels = []

# 在评估代码中添加阈值调整
def predict_with_threshold(outputs, thresholds=[0.25, 0.50, 0.25]):
    probs = torch.softmax(outputs, dim=1)
    # 将概率除以阈值
    adjusted_probs = probs.clone()
    for i, t in enumerate(thresholds):
        adjusted_probs[:, i] = probs[:, i] / t
    # 返回调整后的预测
    return torch.argmax(adjusted_probs, dim=1)

# 在评估循环中使用
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        predicted = predict_with_threshold(outputs)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

# 11. 计算并显示评估指标
class_names = ['下跌 (0)', '稳定 (1)', '上涨 (2)']
print("\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix_seq.png')
plt.show()
    


# 计算总体准确率
accuracy = accuracy_score(all_labels, all_preds)
print(f"总体准确率: {accuracy*100:.2f}%")
from sklearn.metrics import f1_score, balanced_accuracy_score
print(f"F1-macro: {f1_score(all_labels, all_preds, average='macro')}")
print(f"平衡准确率: {balanced_accuracy_score(all_labels, all_preds)}")