import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc , f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
import os
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
train_model = False # 是否训练模型
epochs = 20
target_label = 'label_20'
# 设置支持中文的字体
batch_size = 1024 
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置图形风格
sns.set(style="whitegrid")
base_lr = 0.0005 * (batch_size / 256)**0.5  # 根据批量大小调整

# 1. 加载数据
base_path = r'./train_set/train_set'
files = [
    f"snapshot_sym{j}_date{i}_{session}.csv"
    for j in range(0, 5)  # sym_1 到 sym_5
    for i in range(0,101)  # date_0 到 date_7
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

# 显示数据基本信息
print("\n数据前几行：")
print(df.head())
print("\n数据基本信息：")
print(df.info())
print("\n数据统计描述：")
print(df.describe())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import math

# 设置随机种子，确保结果可复现
torch.manual_seed(777)
np.random.seed(777)

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置图形风格
sns.set(style="whitegrid")

# # 1. 加载数据
# df = pd.read_csv('merged_data.csv')
# print("原始数据形状:", df.shape)
# print(df.head())

# 数据预处理
# 检查缺失值
print("\n检查缺失值:")
print(df.isnull().sum())

# 删除缺失值
df = df.dropna()
print("清理后数据形状:", df.shape)

# 2. 特征工程 - 添加常用因子
# 计算滑动平均
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
            
            # 移动标准差(波动率)
            temp_df[f'{col}_std5'] = temp_df[col].rolling(window=5).std()
            
            # 动量指标
            temp_df[f'{col}_mom5'] = temp_df[col].diff(5)
            
            # 相对于前2,4,7个时间点的价格变化量
            temp_df[f'{col}_change2'] = temp_df[col].diff(2)
            temp_df[f'{col}_change4'] = temp_df[col].diff(4)
            temp_df[f'{col}_change7'] = temp_df[col].diff(7)
            
            # 相对于前2,4,7个时间点的价格变化百分比
            temp_df[f'{col}_pct_change2'] = temp_df[col].pct_change(2)
            temp_df[f'{col}_pct_change4'] = temp_df[col].pct_change(4)
            temp_df[f'{col}_pct_change7'] = temp_df[col].pct_change(7)
        
        # 计算买卖压力指标
        temp_df['buy_pressure'] = temp_df['n_bsize1'] / (temp_df['n_bsize1'] + temp_df['n_asize1'])
        temp_df['sell_pressure'] = temp_df['n_asize1'] / (temp_df['n_bsize1'] + temp_df['n_asize1'])
        
        # # 计算买卖压力比
        # temp_df['pressure_ratio'] = temp_df['buy_pressure'] / temp_df['sell_pressure']
        
        # 计算买卖价差
        temp_df['price_spread'] = temp_df['n_ask1'] - temp_df['n_bid1']
        
        # 计算量价相关性
        temp_df['vol_price_corr'] = (temp_df['amount_delta'] * temp_df['n_close']).rolling(window=5).mean()
        
        # 计算买卖量比
        temp_df['bid_ask_volume_ratio'] = temp_df['n_bsize1'] / temp_df['n_asize1']
        
        # 计算总买单和总卖单
        temp_df['total_bid_volume'] = temp_df['n_bsize1'] + temp_df['n_bsize2'] + temp_df['n_bsize3'] + temp_df['n_bsize4'] + temp_df['n_bsize5']
        temp_df['total_ask_volume'] = temp_df['n_asize1'] + temp_df['n_asize2'] + temp_df['n_asize3'] + temp_df['n_asize4'] + temp_df['n_asize5']
        
        # 计算总买单和总卖单的变化量
        temp_df['total_bid_volume_change'] = temp_df['total_bid_volume'].diff()
        temp_df['total_ask_volume_change'] = temp_df['total_ask_volume'].diff()
        temp_df['total_bid_volume_change2'] = temp_df['total_bid_volume'].diff(2)
        temp_df['total_ask_volume_change2'] = temp_df['total_ask_volume'].diff(2)
        temp_df['total_bid_volume_change4'] = temp_df['total_bid_volume'].diff(4)
        temp_df['total_ask_volume_change4'] = temp_df['total_ask_volume'].diff(4)
        temp_df['total_bid_volume_change7'] = temp_df['total_bid_volume'].diff(7)
        temp_df['total_ask_volume_change7'] = temp_df['total_ask_volume'].diff(7)
        
        # 总交易量不平衡
        temp_df['volume_imbalance'] = (temp_df['total_bid_volume'] - temp_df['total_ask_volume']) / (temp_df['total_bid_volume'] + temp_df['total_ask_volume'])
        
        # 将处理后的数据添加到结果列表
        result_dfs.append(temp_df)
    
    # 合并所有处理后的数据
    result_df = pd.concat(result_dfs)
    
    # 删除生成的NaN值
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
print("添加的新特征:", [col for col in df.columns if col not in ['date', 'sym', 'time','label_5', 'label_10', 'label_20', 'label_40', 'label_60']])
# 检查标签分布
label_counts = df[target_label].value_counts()
print(f"\n标签分布:\n{label_counts}")
print(f"标签比例:\n{df[target_label].value_counts(normalize=True)}")

# 计算每个类别的样本数
class_counts = df[target_label].value_counts()
min_count = min(class_counts)
print(f"最小类别样本数: {min_count}")

# 对数据进行平衡处理
def balance_dataset(df, label_col, random_state=42):
    balanced_dfs = []
    # 获取每个类别的数据
    class_dfs = {i: df[df[label_col] == i] for i in range(3)}
    # 找出最小类别的样本数量
    min_samples = min(len(df_class) for df_class in class_dfs.values())
    # 调整为稍微更大的数量，使各类别之间更加平衡
    target_samples = int(min_samples * 1.5)
    
    # 对每个类别进行处理
    for class_label, class_df in class_dfs.items():
        if len(class_df) < target_samples:
            # 对少数类进行过采样
            resampled = class_df.sample(n=target_samples, replace=True, random_state=random_state)
        else:
            # 对多数类进行欠采样
            resampled = class_df.sample(n=target_samples, replace=False, random_state=random_state)
        balanced_dfs.append(resampled)
    
    # 合并并打乱平衡后的数据
    balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return balanced_df

# # 应用平衡处理
# balanced_df = balance_dataset(df, target_label)
# print(f"平衡后数据形状: {balanced_df.shape}")
# print(f"平衡后标签分布:\n{balanced_df[target_label].value_counts()}")
# print(f"平衡后标签比例:\n{balanced_df[target_label].value_counts(normalize=True)}")

# # 使用平衡后的数据集替换原始数据集
# df = balanced_df
# print("平衡后数据形状:", df.shape, "数据描述:", df.describe())



# # 检查数据是否适合训练和是否需要归一化
# print("\n检查数据是否适合训练:")

# 检查数值特征的分布情况
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
print(f"数值型特征数量: {len(numeric_features)}")

# # 检查是否有极端值
# print("\n检查特征的统计描述:")
# stats = df[numeric_features].describe().T
# stats['range'] = stats['max'] - stats['min']
# large_range_features = stats[stats['range'] > 1000]['range']
# if not large_range_features.empty:
#     print(f"发现{len(large_range_features)}个范围过大的特征:")
#     print(large_range_features.sort_values(ascending=False).head(10))
#     print("这些特征可能需要归一化处理")
# else:
#     print("没有发现范围过大的特征")

# # 检查特征间的相关性
# print("\n检查特征间的相关性:")
# corr_matrix = df[numeric_features].corr().abs()
# high_corr_features = {}
# for i in range(len(corr_matrix.columns)):
#     for j in range(i+1, len(corr_matrix.columns)):
#         if corr_matrix.iloc[i, j] > 0.95:
#             col_i = corr_matrix.columns[i]
#             col_j = corr_matrix.columns[j]
#             high_corr_features[(col_i, col_j)] = corr_matrix.iloc[i, j]

# if high_corr_features:
#     print(f"发现{len(high_corr_features)}对高度相关的特征(相关系数>0.95):")
#     sorted_corrs = sorted(high_corr_features.items(), key=lambda x: x[1], reverse=True)
#     for (col1, col2), corr in sorted_corrs[:10]:
#         print(f"{col1} 和 {col2}: {corr:.4f}")
#     print("考虑移除冗余特征以提高模型效率")
# else:
#     print("没有发现高度相关的特征对")

# # 检查特征方差是否接近于零
# print("\n检查低方差特征:")
# variances = df[numeric_features].var()
# low_var_features = variances[variances < 0.01]
# if not low_var_features.empty:
#     print(f"发现{len(low_var_features)}个低方差特征:")
#     print(low_var_features.sort_values().head(10))
#     print("这些特征可能对模型贡献较小，考虑移除")
# else:
#     print("没有发现低方差特征")

# # 检查各个数值特征的分布情况
# plt.figure(figsize=(15, 10))
# plt.suptitle("数值特征分布直方图", fontsize=16)
# sample_features = np.random.choice(numeric_features, min(12, len(numeric_features)), replace=False)
# for i, feature in enumerate(sample_features):
#     plt.subplot(3, 4, i+1)
#     plt.hist(df[feature], bins=50)
#     plt.title(feature)
#     plt.tight_layout()
# plt.subplots_adjust(top=0.9)
# plt.savefig('feature_distributions.png')
# plt.close()

# print("\n随机特征分布图已保存至'feature_distributions.png'")
# 3. 按日期进行归一化
def normalize_by_date(df, date_column='date'):
    """
    按日期分组对数据进行标准化，使用向量化操作提高效率
    """
    # 数值型列
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # 排除标签列
    numeric_cols = [col for col in numeric_cols if not col.startswith('label_')]
    # 排除日期和系统编号列
    numeric_cols = [col for col in numeric_cols if col not in ['date', 'sym','sys_0', 'sys_1', 'sys_2', 'sys_3', 'sys_4']]
    # 创建结果DataFrame的副本
    result_df = df.copy()
    
    # 将所有数值列转换为float64类型
    result_df[numeric_cols] = result_df[numeric_cols].astype('float64')
    
    # 先替换无穷值为NaN
    result_df[numeric_cols] = result_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # 使用中位数填充NaN值
    result_df[numeric_cols] = result_df[numeric_cols].fillna(result_df[numeric_cols].median())
    
    # 使用transform方法按组计算均值和标准差并直接应用标准化
    # 这会一次性处理所有日期组，无需显式循环
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
# 保存处理后的数据
# print("保存处理后的数据...")
# output_path = 'processed_data.csv'
# df_normalized.to_csv(output_path, index=False)
# print(f"数据已保存到 {output_path}")

feature_cols = [col for col in df_normalized.columns 
                if col not in ['date', 'time', 'sym', 'label_5', 'label_10', 'label_20', 'label_40', 'label_60']]

# 特征和标签
X = df_normalized[feature_cols].values
y = df_normalized[target_label].astype(int).values  # 确保标签是整数类型

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=777, stratify=y)

# 基于时间序列的训练/测试分割
print("\n执行基于日期的时序分割...")

# # 虽然特征中排除了日期，但原始df_normalized数据框应该仍然包含'date'列
if 'date' not in df_normalized.columns:
    print("警告: 数据中没有'date'列, 回退到随机分割!")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=777777777, stratify=y)
else:
    # 获取唯一日期并排序
    unique_dates = sorted(df_normalized['date'].unique())
    
    if len(unique_dates) < 2:
        print(f"警告: 找到的唯一日期数量不足 ({len(unique_dates)}), 回退到随机分割!")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=777777777, stratify=y)
    else:
        # 使用前80%的日期作为训练集，后20%作为测试集
        split_idx = int(len(unique_dates) * 0.8)
        train_dates = unique_dates[:split_idx]
        test_dates = unique_dates[split_idx:]
        
        print(f"训练集日期范围: {train_dates[0]} 到 {train_dates[-1]}, 共{len(train_dates)}个日期")
        print(f"测试集日期范围: {test_dates[0]} 到 {test_dates[-1]}, 共{len(test_dates)}个日期")
        
        # 创建布尔索引掩码，表示每行是否属于训练集
        train_indices = df_normalized['date'].isin(train_dates)
        test_indices = df_normalized['date'].isin(test_dates)
        
        # 应用掩码分割特征和标签
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        
        print(f"训练集大小: {X_train.shape[0]}个样本")
        print(f"测试集大小: {X_test.shape[0]}个样本")
        print(f"训练集标签分布: {np.bincount(y_train)}")
        print(f"测试集标签分布: {np.bincount(y_test)}")
# # 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 5. 定义Transformer模型
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_linear = nn.Linear(d_model, num_classes)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.src_mask = None

    def forward(self, src):
        # src shape: [batch_size, feature_dim]
        # 增加序列维度，将每个特征视为序列中的一个元素
        src = src.unsqueeze(0)  # [1, batch_size, feature_dim]
        src = self.input_linear(src) * math.sqrt(self.d_model)  # [1, batch_size, d_model]
        src = self.pos_encoder(src)  # [1, batch_size, d_model]
        output = self.transformer_encoder(src)  # [1, batch_size, d_model]
        output = output.mean(dim=0)  # [batch_size, d_model]
        output = self.dropout(output)
        output = self.output_linear(output)  # [batch_size, num_classes]
        return output

# 6. 初始化模型
input_dim = len(feature_cols)  # 特征维度
d_model = 256   # 模型维度
nhead = 8  # 多头注意力中的头数
num_layers = 6  # Transformer编码器层数
dim_feedforward = 1024  # 前馈网络的维度
num_classes = 3  # 类别数：下降(0)、稳定(1)、上升(2)
dropout = 0.4  # Dropout率

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = TransformerModel(input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout).to(device)

# 检查是否有预训练模型，有则加载
model_path = f'models/best_transformer_model_{target_label}.pth'
if os.path.exists(model_path):
    print(f"加载已有模型: {model_path}")
    model.load_state_dict(torch.load(model_path))
else:
    print("没有找到预训练模型，将从头开始训练")
    
if train_model == True:
    # 7. 训练模型
    criterion = nn.CrossEntropyLoss()
    

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.03)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=base_lr*1.5,  # 最大学习率通常是基础学习率的3-5倍
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,     # 使用10%的迭代进行预热
        div_factor=25,
        final_div_factor=1e4      # 初始学习率 = max_lr/25
    )
    # 创建保存模型的目录
    if not os.path.exists('models'):
        os.makedirs('models')

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            # 添加在这里 - 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss = val_loss / len(test_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型（基于准确率）
        if val_acc > getattr(model, 'best_val_acc', 0):
            model.best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Epoch {epoch+1}/{epochs} - 保存新的最佳模型, 验证准确率: {val_acc:.2f}%")
        
        print(f"Epoch {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%, 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")

    # 8. 绘制训练过程
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
    plt.savefig('training_history.png')
    plt.show()

# 9. 在测试集上评估最佳模型
model.load_state_dict(torch.load(model_path))
model.eval()

all_preds = []
all_labels = []

# 在这里放置层次分类逻辑
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # 第一步：判断是否稳定
        outputs = model(batch_X)
        probas = F.softmax(outputs, dim=1)
        
        # 使用层次分类逻辑：
        # 1. 先判断是否稳定
        is_stable = probas[:, 1] > 0.4
        
        # 2. 对非稳定样本再判断上涨/下跌
        non_stable = ~is_stable
        up_vs_down = probas[non_stable, 2] > probas[non_stable, 0]
        
        # 3. 组合最终预测
        final_preds = torch.ones_like(batch_y)  # 默认为稳定(1)
        final_preds[non_stable] = torch.where(up_vs_down, 
                                           torch.tensor(2, device=device), 
                                           torch.tensor(0, device=device))
        
        # 收集预测和标签用于最终评估
        all_preds.extend(final_preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

# 10. 计算并显示评估指标
class_names = ['Down (0)', 'Stable (1)', 'Up (2)']
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
f1_macro = f1_score(all_labels, all_preds, average='macro')
print(f"宏平均F1分数: {f1_macro:.4f}")
# 绘制混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# 11. 特征重要性分析
def get_feature_importance(model, test_loader, feature_names, device):
    model.eval()
    importances = np.zeros(len(feature_names))
    
    # 基于置换特征的方法来计算特征重要性
    base_accuracy = accuracy_score(all_labels, all_preds)
    
    for i in range(len(feature_names)):
        # 创建一个从原始数据集复制的数据集，但将第i个特征打乱
        X_test_permuted = X_test.copy()
        np.random.shuffle(X_test_permuted[:, i])
        
        # 转换为PyTorch张量
        permuted_tensor = torch.FloatTensor(X_test_permuted).to(device)
        
        # 预测
        preds = []
        with torch.no_grad():
            for j in range(0, len(permuted_tensor), batch_size):
                end = min(j + batch_size, len(permuted_tensor))
                batch = permuted_tensor[j:end]
                outputs = model(batch)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
        
        # 计算打乱后的准确率
        permuted_accuracy = accuracy_score(y_test, preds)
        
        # 特征重要性 = 原始准确率 - 打乱后的准确率
        importances[i] = base_accuracy - permuted_accuracy
    
    return importances

# 计算特征重要性
feature_names = feature_cols
importances = get_feature_importance(model, test_loader, feature_names, device)

# 按重要性排序
indices = np.argsort(importances)[::-1]
sorted_features = [feature_names[i] for i in indices]
sorted_importances = importances[indices]
# 将特征重要性保存到文本文件
with open(f'feature_importance_{target_label}.txt', 'w') as f:
    f.write("Feature Importance Ranking:\n")
    f.write("==========================\n\n")
    for i in range(len(sorted_features)):
        f.write(f"{i+1}. {sorted_features[i]}: {sorted_importances[i]:.6f}\n")
    
    # 添加一些基本统计信息
    f.write("\n\nSummary Statistics:\n")
    f.write("=================\n")
    f.write(f"Mean Importance: {np.mean(importances):.6f}\n")
    f.write(f"Median Importance: {np.median(importances):.6f}\n")
    f.write(f"Max Importance: {np.max(importances):.6f} ({feature_names[np.argmax(importances)]})\n")
    f.write(f"Min Importance: {np.min(importances):.6f} ({feature_names[np.argmin(importances)]})\n")

print(f"Feature importance saved to 'feature_importance_{target_label}.txt'")
# 绘制特征重要性
plt.figure(figsize=(12, 8))
plt.bar(range(len(sorted_importances[:20])), sorted_importances[:20])
plt.xticks(range(len(sorted_importances[:20])), sorted_features[:20], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Top 20 Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\nTop 10 Most Important Features:")
for i in range(10):
    print(f"{sorted_features[i]}: {sorted_importances[i]:.4f}")