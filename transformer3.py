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
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
target_labels = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
seq_length = 60

train_model = False # 是否训练模型
train_model_hierarchical =  False # 是否训练分层模型
show_pic = False # 是否显示图片


epochs = 20
# target_labels = target_labels[0]  # 选择目标标签
target_label =[ target_labels[0]]
patience = 5 # 早停耐心值
class_weights = torch.tensor([2.5, 0.6, 2.6], device=device)
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
        # x shape: [batch_size, seq_len, input_dim]
        
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

# 分层Transformer实现
class HierarchicalTransformer:
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        """初始化分层Transformer模型"""
        # 第一层模型：判断是否稳定
        self.stability_model = TransformerModel(
            input_dim, d_model, nhead, num_layers, dim_feedforward, 2, dropout).to(device)
        
        # 第二层模型：判断上涨或下跌
        self.direction_model = TransformerModel(
            input_dim, d_model, nhead, num_layers, dim_feedforward, 2, dropout).to(device)
        
        # 模型路径
        self.stability_model_path = f'./models/stability_model_{target_label}.pth'
        self.direction_model_path = f'./models/direction_model_{target_label}.pth'
        
        # 尝试加载已有模型
        self.load_models()
    
    def load_models(self):
        """加载预训练模型（如果存在）"""
        if os.path.exists(self.stability_model_path):
            print(f"加载稳定性模型: {self.stability_model_path}")
            self.stability_model.load_state_dict(torch.load(self.stability_model_path))
        
        if os.path.exists(self.direction_model_path):
            print(f"加载方向性模型: {self.direction_model_path}")
            self.direction_model.load_state_dict(torch.load(self.direction_model_path))
    
    def save_models(self):
        """保存当前模型"""
        torch.save(self.stability_model.state_dict(), self.stability_model_path)
        torch.save(self.direction_model.state_dict(), self.direction_model_path)
    
    def train(self, train_loader, val_loader, epochs=20):
        """训练分层模型"""
        # 准备第一层模型数据：转换标签(1->1, 0或2->0)
        stability_train_data = self._prepare_stability_data(train_loader)
        stability_val_data = self._prepare_stability_data(val_loader)
        
        # 准备第二层模型数据：仅保留非稳定样本(0->0, 2->1)
        direction_train_data = self._prepare_direction_data(train_loader)
        direction_val_data = self._prepare_direction_data(val_loader)
        
        # 训练第一层模型（稳定性判断）
        print("\n开始训练稳定性模型...")
        self._train_single_model(
            self.stability_model, 
            stability_train_data, 
            stability_val_data, 
            epochs, 
            self.stability_model_path
        )
        
        # 训练第二层模型（方向判断）
        print("\n开始训练方向性模型...")
        self._train_single_model(
            self.direction_model, 
            direction_train_data, 
            direction_val_data, 
            epochs, 
            self.direction_model_path
        )
    
    def _prepare_stability_data(self, data_loader):
        """准备稳定性模型的数据（二分类：稳定vs非稳定）- 显存优化版"""
        class StabilityDataset:
            def __init__(self, data_loader):
                self.data_loader = data_loader
                self.length = len(data_loader)
            
            def __len__(self):
                return self.length
            
            def __iter__(self):
                for batch_X, batch_y in self.data_loader:
                    # 转换标签：1保持为1（稳定），0和2变为0（不稳定）
                    new_y = torch.zeros_like(batch_y)
                    new_y[batch_y == 1] = 1  # 稳定类别
                    yield batch_X, new_y
                    # 手动触发垃圾回收
                    del new_y
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        return StabilityDataset(data_loader)
    
    def _prepare_direction_data(self, data_loader):
        """准备方向性模型的数据（二分类：上涨vs下跌，仅使用非稳定样本）- 显存优化版"""
        class DirectionDataset:
            def __init__(self, data_loader):
                self.data_loader = data_loader
                # 需要计算实际长度
                self.length = 0
                for _, batch_y in data_loader:
                    mask = batch_y != 1
                    if mask.sum() > 0:
                        self.length += 1
            
            def __len__(self):
                return self.length
            
            def __iter__(self):
                for batch_X, batch_y in self.data_loader:
                    # 筛选非稳定样本
                    mask = batch_y != 1
                    if mask.sum() > 0:  # 确保有非稳定样本
                        filtered_X = batch_X[mask]
                        filtered_y = batch_y[mask]
                        
                        # 转换标签：0->0（下跌），2->1（上涨）
                        new_y = torch.zeros_like(filtered_y)
                        new_y[filtered_y == 2] = 1  # 上涨类别
                        
                        yield filtered_X, new_y
                        
                        # 手动释放不需要的张量
                        del filtered_X, filtered_y, new_y, mask
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        
        return DirectionDataset(data_loader)
    
    def _train_single_model(self, model, train_data, val_data, epochs, model_path):
        """训练单个模型 - 显存优化版"""
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        
        # 检查是否已有模型
        existing_val_acc = 0
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            model.load_state_dict(torch.load(model_path))
            # 检查现有模型性能
            model.eval()
            with torch.no_grad():
                val_correct = 0
                val_total = 0
                for batch_X, batch_y in val_data:
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
                    
                    # 释放显存
                    del outputs, predicted
                    torch.cuda.empty_cache()
                    
                existing_val_acc = 100 * val_correct / val_total if val_total > 0 else 0
                print(f"Existing model validation accuracy: {existing_val_acc:.2f}%")
                if existing_val_acc > 80:  # 如果模型已经表现良好
                    print(f"Existing model performs well, skipping training")
                    return
                    
        best_val_acc = existing_val_acc
        no_improve = 0
        patience = 5
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # 使用生成器风格的数据集
            for batch_X, batch_y in train_data:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                # 释放显存
                del outputs, loss, predicted
                torch.cuda.empty_cache()
            
            train_loss = total_loss / len(train_data)
            train_acc = 100 * correct / total if total > 0 else 0
            
            # 验证阶段
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_data:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    
                    # 释放显存
                    del outputs, loss, predicted
                    torch.cuda.empty_cache()
            
            val_loss = val_loss / len(val_data) if len(val_data) > 0 else float('inf')
            val_acc = 100 * correct / total if total > 0 else 0
            
            # 更新学习率
            scheduler.step(val_acc)
            
            # 打印结果
            print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, '
                f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print(f"模型已保存: {model_path}")
                no_improve = 0
            else:
                no_improve += 1
                
            # 早停
            if no_improve >= patience:
                print(f"早停: {patience} 轮无改善")
                break
    
    def predict(self, X):
        """使用分层模型进行预测"""
        self.stability_model.eval()
        self.direction_model.eval()
        with torch.no_grad():
            # 第一步：预测是否稳定
            stability_output = self.stability_model(X)
            stability_pred = torch.argmax(stability_output, dim=1)
            
            # 初始化最终预测结果
            final_pred = torch.ones_like(stability_pred)  # 默认值为1（稳定）
            
            # 第二步：对于不稳定样本，预测上涨或下跌
            unstable_mask = (stability_pred == 0)
            if torch.any(unstable_mask):
                unstable_X = X[unstable_mask]
                direction_output = self.direction_model(unstable_X)
                direction_pred = torch.argmax(direction_output, dim=1)
                
                # 合并结果：0=下跌，2=上涨
                final_pred[unstable_mask] = torch.where(direction_pred == 1, 
                                                    torch.tensor(2, device=device), 
                                                    torch.tensor(0, device=device))
            
            return final_pred

# 评估分层模型函数
def evaluate_hierarchical_model(hierarchical_model, data_loader, device):
    """评估分层模型性能"""
    hierarchical_model.stability_model.eval()
    hierarchical_model.direction_model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            # 数据已经在GPU上，无需再次转移
            predictions = hierarchical_model.predict(batch_X)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return accuracy, report, conf_matrix, all_preds, all_labels

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
    train_idx = int(n * 0.94)
    val_idx = int(n * 0.97)
    
    # 标准化特征
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:val_idx]
    test_df = df.iloc[val_idx:]
    # if not train_model_hierarchical and train_model:
    #     # 仅使用非稳定样本进行训练
        
    print(f"训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}, 测试集大小: {len(test_df)}")
    
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
    
    return train_loader, val_loader, test_loader, len(feature_cols)

def ensemble_predict(X, standard_model, hierarchical_model, stable_threshold=0.75):
    # 获取标准模型预测和置信度
    standard_output = standard_model(X)
    standard_probs = F.softmax(standard_output, dim=1)
    standard_pred = torch.argmax(standard_probs, dim=1)
    
    # 获取分层模型预测
    hierarchical_pred = hierarchical_model.predict(X)
    
    # 创建集成预测
    ensemble_pred = standard_pred.clone()
    
    # 当标准模型对稳定类预测置信度不高时，采用分层模型预测
    uncertain_mask = (standard_probs[:, 1] < stable_threshold) | (standard_pred != 1)
    ensemble_pred[uncertain_mask] = hierarchical_pred[uncertain_mask]
    
    return ensemble_pred

def main():
    # 1. 加载数据 - 保持原有代码
    start_time = time.time()
    print("开始加载数据文件...")
    

    # 合并所有文件
    df_list = []
    for file in files:
        file_path = os.path.join(base_path, file)
        if os.path.exists(file_path):
            temp_df = pd.read_csv(file_path)
            df_list.append(temp_df)
            # print(f"加载文件: {file_path}，数据形状: {temp_df.shape}")
        else:
            print(f"文件 {file} 不存在，跳过。")

    # 合并数据集
    print("合并所有数据文件...")
    df = pd.concat(df_list, ignore_index=True)
    
    # 提取特征列（除标签和日期外）
    print("提取特征列...")
    feature_cols = [col for col in df.columns if col not in ['date','time','label_5', 'label_10', 'label_20', 'label_40', 'label_60']]
    
    # 使用优化的数据加载函数
    print(f"开始数据预处理和GPU加载，使用目标标签: {target_label}")
    train_loader, val_loader, test_loader, input_dim = preprocess_and_load_to_gpu(df, feature_cols, target_label, seq_length)
    
    data_prep_time = time.time() - start_time
    print(f"数据准备阶段总耗时: {data_prep_time:.2f}秒")
    
    # 模型参数
    d_model = 128  # 模型维度
    nhead = 8  # 多头注意力头数
    num_layers = 3  # Transformer编码器层数
    dim_feedforward = 256  # 前馈网络维度
    num_classes = 3  # 类别数：下跌、稳定、上涨
    
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
    
    # 初始化分层Transformer模型
    print("初始化分层Transformer模型...")
    hierarchical_model = HierarchicalTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    model_path = f'./models/standard_model_{target_label}.pth'
    if os.path.exists(model_path) and not train_model:
        print(f"加载已有标准模型: {model_path}")
        standard_model.load_state_dict(torch.load(model_path, weights_only=True))
    if train_model:
        print("训练标准Transformer模型...")
        train_start = time.time()
        
        # 训练标准模型
        optimizer = optim.AdamW(standard_model.parameters(), lr=base_lr, weight_decay=0.05)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',           # 基于验证准确率最大化
            factor=0.5,           # 每次减半
            patience=4,           # 3轮无改善则降低学习率
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
            epoch_start = time.time()
            standard_model.train()
            total_loss = 0
            correct = 0
            total = 0

            # 训练循环
            for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
                # 数据已经在GPU上，无需再次转移
                
                optimizer.zero_grad()
                outputs = standard_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
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
                    # 数据已经在GPU上，无需再次转移
                    
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
            if val_acc > best_val_acc*0.998:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(standard_model.state_dict(), f'./models/standard_model_{target_label}.pth')
                    print(f"保存最佳标准模型，验证准确率: {val_acc:.2f}%")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            # 早停
            if no_improve_epochs >= patience:
                print(f"早停: {patience} 轮无改善")
                break
        
        standard_train_time = time.time() - train_start
        print(f"标准模型训练完成，总耗时: {standard_train_time:.2f}秒")
        pict_dir = './figures'
        if not os.path.exists(pict_dir):
            os.makedirs(pict_dir)
        if show_pic:
        # Plot training history
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title('Loss Curves')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(train_accs, label='Training Accuracy')
            plt.plot(val_accs, label='Validation Accuracy')
            plt.title('Accuracy Curves')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.legend()

            plt.tight_layout()
            plt.savefig(f'{pict_dir}/training_history_{target_label}.png')
            # plt.show()

    # 训练分层模型
    if train_model_hierarchical:
        print("\n训练分层Transformer模型...")
        hierarchical_train_start = time.time()
        hierarchical_model.train(train_loader, val_loader, epochs)
        hierarchical_train_time = time.time() - hierarchical_train_start
        print(f"分层模型训练完成，总耗时: {hierarchical_train_time:.2f}秒")

        
        
    # Load best standard model
    print("Loading best standard model for evaluation...")
    standard_model.load_state_dict(torch.load(f'./models/standard_model_{target_label}.pth'))
    
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
    if show_pic:
        # Plot standard model confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(std_conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Standard Transformer Model Confusion Matrix - {target_label}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{pict_dir}/training_history_{target_label}.png')
        # plt.show()
    """

    要使用的模型
    
    """
    # Evaluate hierarchical model
    print("\nEvaluating Hierarchical Transformer Model...")
    hier_eval_start = time.time()
    hier_accuracy, hier_report, hier_conf_matrix, hier_all_preds, hier_all_labels = evaluate_hierarchical_model(
    hierarchical_model, test_loader, device
    )
    hier_eval_time = time.time() - hier_eval_start
    print(f"Hierarchical model evaluation completed, time: {hier_eval_time:.2f} seconds")
    
    print("\nHierarchical Transformer Model Evaluation:")
    print(f"Test Accuracy: {hier_accuracy:.4f}")
    print("\nClassification Report:")
    print(hier_report)
    if show_pic:
    # Plot hierarchical model confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(hier_conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Hierarchical Transformer Model Confusion Matrix - {target_label}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{pict_dir}/training_history_{target_label}.png')
        # plt.show()
    
    # Compare model performance
    print("\nModel Performance Comparison:")
    print(f"Standard Transformer Accuracy: {std_accuracy:.4f}")
    print(f"Hierarchical Transformer Accuracy: {hier_accuracy:.4f}")
    print(f"Performance Improvement: {(hier_accuracy - std_accuracy) * 100:.2f}%")
    
    # Generate performance comparison for each class
    class_names = ['Down', 'Stable', 'Up']
    std_class_report = classification_report(std_all_labels, std_all_preds, output_dict=True)
    hier_class_report = classification_report(hier_all_labels, hier_all_preds, output_dict=True)
    
    # Create class performance comparison table
    class_comparison = []
    for i, class_name in enumerate(class_names):
        std_f1 = std_class_report[str(i)]['f1-score']
        hier_f1 = hier_class_report[str(i)]['f1-score']
        improvement = (hier_f1 - std_f1) * 100
        class_comparison.append({
            'Class': class_name,
            'Standard Model F1': std_f1,
            'Hierarchical Model F1': hier_f1,
            'Improvement (%)': improvement
        })
    
    class_df = pd.DataFrame(class_comparison)
    print("\nClass Performance Comparison:")
    print(class_df)
    # Create a DataFrame with results from standard and hierarchical models
    calls_df = pd.DataFrame({
        'True_Label': std_all_labels,
        'Standard_Pred': std_all_preds,
        'Hierarchical_Pred': hier_all_preds
    })

    # Save results to txt files
    results_dir = f'./results'
    results_dir_pred = f'./results/predictions'
    os.makedirs(results_dir_pred, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Save the calls dataframe
    calls_df.to_csv(f'{results_dir_pred}/prediction_results_{target_label}.txt', sep='\t', index=False)

    # Save standard model results
    with open(f'{results_dir}/standard_model_results_{target_label}.txt', 'w') as f:
        f.write(f"Accuracy: {std_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(str(std_report))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(std_conf_matrix))

    # Save hierarchical model results
    with open(f'{results_dir}/hierarchical_model_results.txt_{target_label}', 'w') as f:
        f.write(f"Accuracy: {hier_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(str(hier_report))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(hier_conf_matrix))

    print(f"\nResults saved to {results_dir}")
    # Plot class performance comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(class_names))
    width = 0.35
    if show_pic:
        plt.bar(x - width/2, class_df['Standard Model F1'], width, label='Standard Model')
        plt.bar(x + width/2, class_df['Hierarchical Model F1'], width, label='Hierarchical Model')
        
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Comparison Between Models by Class')
        plt.xticks(x, class_names)
        plt.legend()
        plt.savefig(f'model_comparison_{target_label}.png')
        plt.show()
    # 在main函数中现有评估标准模型和分层模型的代码后添加：

    # Print GPU memory usage
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
        print(f"Cached Memory After Cleanup: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

if __name__ == "__main__":
    for lab in target_labels:
        target_label = lab
        main()
