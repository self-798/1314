import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

# 数据加载路径设置
base_path = r'./train_set/train_set'
files = [
    f"snapshot_sym{j}_date{i}_{session}.csv"
    for j in range(0, 5)  # sym_0 到 sym_4
    for i in range(0, 102)  # date_0 到 date_100
    for session in ['am', 'pm']
]

# 合并所有文件
def load_data(files):
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
    print(f"合并后数据形状: {df.shape}")
    return df

# 数据预处理
def preprocess_data(df, target_label='label_20'):
    # 删除缺失值
    df = df.dropna()
    
    # 创建衍生特征
    # 价格差异特征
    df['bid_ask_spread'] = df['n_ask1'] - df['n_bid1']
    df['mid_price'] = (df['n_ask1'] + df['n_bid1']) / 2
    df['price_range'] = df['n_ask1'] - df['n_bid1']
    
    # 量比特征
    df['volume_imbalance'] = (df['n_asize1'] - df['n_bsize1']) / (df['n_asize1'] + df['n_bsize1'] + 0.001)
    df['total_volume'] = df['n_asize1'] + df['n_bsize1']
    
    # 多级深度特征
    # 计算价格阶梯
    df['ask_price_slope'] = (df['n_ask2'] - df['n_ask1']) / (df['n_ask1'] + 0.001)
    df['bid_price_slope'] = (df['n_bid1'] - df['n_bid2']) / (df['n_bid1'] + 0.001)
    # 计算量阶梯
    df['ask_size_slope'] = df['n_asize2'] / (df['n_asize1'] + 0.001)
    df['bid_size_slope'] = df['n_bsize2'] / (df['n_bsize1'] + 0.001)
    
    # 时间特征
    if 'time' in df.columns:
        df['time_numeric'] = pd.to_datetime(df['time']).dt.hour * 60 + pd.to_datetime(df['time']).dt.minute
        # 市场阶段特征
        df['morning_session'] = (df['time_numeric'] < 720).astype(int)  # 上午交易时段
    
    # OHLC特征
    df['high_low_diff'] = df['n_high'] - df['n_low']
    df['close_open_diff'] = df['n_close'] - df['n_open']
    df['close_mid_diff'] = df['n_close'] - df['n_midprice']
    
    # 按日期归一化数值特征
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if not col.startswith('label_')]
    
    # 按日期分组标准化
    grouped = df.groupby('date')
    result_dfs = []
    
    for date, group in grouped:
        # 计算均值和标准差
        means = group[numeric_cols].mean()
        stds = group[numeric_cols].std().replace(0, 1)
        
        # 标准化
        normalized = group.copy()
        normalized[numeric_cols] = (group[numeric_cols] - means) / stds
        
        result_dfs.append(normalized)
    
    df_normalized = pd.concat(result_dfs)
    
    # 添加波动率特征
    rolling_window = 10  # 滚动窗口大小
    if len(df) >= rolling_window:
        # 中间价格波动率
        df['mid_price_volatility'] = df['mid_price'].rolling(window=rolling_window).std().fillna(0)
        # 价差波动率
        df['spread_volatility'] = df['bid_ask_spread'].rolling(window=rolling_window).std().fillna(0)
        # 成交量波动率
        df['volume_volatility'] = df['total_volume'].rolling(window=rolling_window).std().fillna(0)
        # 价格波动率
        df['price_volatility'] = df['n_midprice'].rolling(window=rolling_window).std().fillna(0)
    
    # 流动性指标
    df['liquidity_ratio'] = df['total_volume'] / (df['bid_ask_spread'] + 0.001)  # 避免除零
    
    # 订单不平衡指标
    df['order_imbalance'] = (df['n_bsize1'] - df['n_asize1']) / (df['n_bsize1'] + df['n_asize1'] + 0.001)
    
    # 交易压力指标
    df['buy_pressure'] = df['n_bsize1'] * df['n_bid1']
    df['sell_pressure'] = df['n_asize1'] * df['n_ask1']
    df['pressure_ratio'] = df['buy_pressure'] / (df['sell_pressure'] + 0.001)
    
    # 价格动量指标
    df['price_momentum_5'] = df['n_midprice'] - df['n_midprice'].shift(5)
    df['price_momentum_10'] = df['n_midprice'] - df['n_midprice'].shift(10)
    df['price_momentum_5'].fillna(0, inplace=True)
    df['price_momentum_10'].fillna(0, inplace=True)
    
    # 深度指标 - 考虑更多级别的报价
    df['depth_ask_ratio'] = (df['n_asize1'] + df['n_asize2'] + df['n_asize3']) / (df['n_asize1'] + 0.001)
    df['depth_bid_ratio'] = (df['n_bsize1'] + df['n_bsize2'] + df['n_bsize3']) / (df['n_bsize1'] + 0.001)
    
    # 波动率比率
    if len(df) >= 2*rolling_window:
        # 短期波动率与长期波动率比较
        short_term = df['n_midprice'].rolling(window=rolling_window).std().fillna(0)
        long_term = df['n_midprice'].rolling(window=2*rolling_window).std().fillna(0)
        df['volatility_ratio'] = short_term / (long_term + 0.001)
    
    # amount_delta特征
    if 'amount_delta' in df.columns:
        df['amount_delta_norm'] = df['amount_delta'] / (df['total_volume'] + 0.001)
    
    # 准备特征和标签
    feature_cols = [col for col in df_normalized.columns 
                    if col not in ['date', 'time', 'sym', 'label_5', 'label_10', 'label_20', 'label_40', 'label_60']]
    
    X = df_normalized[feature_cols].values
    y = df_normalized[target_label].values
    
    return X, y, feature_cols

# 划分训练集和测试集
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    return X_train, X_test, y_train, y_test

# # 训练并评估多个分类模型
# def train_evaluate_models(X_train, X_test, y_train, y_test, feature_cols):
#     # 确保标签是整数类型
#     y_train = y_train.astype(int)
#     y_test = y_test.astype(int)
    
#     models = {
#         '逻辑回归': LogisticRegression(max_iter=1000, multi_class='multinomial'),
#         'Ridge分类': LogisticRegression(penalty='l2', C=1.0, max_iter=1000, multi_class='multinomial'),
#         'L1分类': LogisticRegression(penalty='l1', C=1.0, solver='saga', max_iter=1000, multi_class='multinomial')
#     }
    
#     results = {}
#     feature_importance = {}
    
#     for name, model in models.items():
#         # 训练模型
#         model.fit(X_train, y_train)
        
#         # 预测
#         y_pred = model.predict(X_test)
        
#         # 评估
#         accuracy = accuracy_score(y_test, y_pred) * 100
#         conf_matrix = confusion_matrix(y_test, y_pred)
#         class_report = classification_report(y_test, y_pred, 
#                                             target_names=['下跌', '稳定', '上涨'])
        
#         results[name] = {
#             'Accuracy': accuracy,
#             'Confusion Matrix': conf_matrix,
#             'Classification Report': class_report
#         }
        
#         # 特征重要性
#         if hasattr(model, 'coef_'):
#             # 对于多分类，coef_形状是 [n_classes, n_features]
#             importances = np.mean(np.abs(model.coef_), axis=0)  # 平均各类别的权重绝对值
#             feature_importance[name] = pd.DataFrame({
#                 '特征': feature_cols,
#                 '重要性': importances
#             }).sort_values(by='重要性', ascending=False)
        
#         # 打印混淆矩阵
#         print(f"\n{name} 混淆矩阵:")
#         print(pd.DataFrame(
#             conf_matrix,
#             index=["实际 下跌", "实际 稳定", "实际 上涨"],
#             columns=["预测 下跌", "预测 稳定", "预测 上涨"]
#         ))
#         print(f"{name} 准确率: {accuracy:.2f}%")
#         print("\n分类报告:")
#         print(class_report)
    
#     return results, feature_importance
    # 训练并评估多个分类模型
def train_evaluate_models(X_train, X_test, y_train, y_test, feature_cols):
    # 确保标签是整数类型
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    # 多分类模型
    print("\n===== 三分类模型 =====")
    multi_model = LogisticRegression(max_iter=5000)
    multi_model.fit(X_train, y_train)
    multi_pred = multi_model.predict(X_test)
    
    multi_accuracy = accuracy_score(y_test, multi_pred) * 100
    multi_conf_matrix = confusion_matrix(y_test, multi_pred)
    multi_report = classification_report(y_test, multi_pred, 
                                       target_names=['下跌', '稳定', '上涨'])
    
    print(f"\n三分类模型 准确率: {multi_accuracy:.2f}%")
    print("混淆矩阵:")
    print(pd.DataFrame(
        multi_conf_matrix,
        index=["实际 下跌", "实际 稳定", "实际 上涨"],
        columns=["预测 下跌", "预测 稳定", "预测 上涨"]
    ))
    print("\n分类报告:")
    print(multi_report)
    
    # 第一级模型：稳定vs非稳定
    print("\n===== 第一级分类：稳定 vs 非稳定 =====")
    y_train_binary = (y_train == 1).astype(int)
    y_test_binary = (y_test == 1).astype(int)
    
    stable_model = LogisticRegression(max_iter=1000)
    stable_model.fit(X_train, y_train_binary)
    stable_pred = stable_model.predict(X_test)
    
    stable_accuracy = accuracy_score(y_test_binary, stable_pred) * 100
    stable_conf_matrix = confusion_matrix(y_test_binary, stable_pred)
    
    print(f"\n稳定性分类 准确率: {stable_accuracy:.2f}%")
    print("混淆矩阵:")
    print(pd.DataFrame(
        stable_conf_matrix,
        index=["实际 非稳定", "实际 稳定"],
        columns=["预测 非稳定", "预测 稳定"]
    ))
    
    # 第二级模型：对于预测为非稳定的样本进行三分类
    print("\n===== 第二级分类：三分类（针对预测为非稳定的样本）=====")
    # 获取预测为非稳定的样本索引
    non_stable_pred_idx = np.where(stable_pred == 0)[0]
    X_second_level = X_test[non_stable_pred_idx]
    y_second_level = y_test[non_stable_pred_idx]
    
    # 训练第二级三分类模型（使用原始训练数据）
    second_level_model = LogisticRegression(max_iter=1000, multi_class='multinomial')
    second_level_model.fit(X_train, y_train)  # 使用完整训练集
    
    # 对预测为非稳定的样本进行第二级预测
    second_level_pred = second_level_model.predict(X_second_level)
    
    # 整合两个模型的预测结果
    final_pred = np.ones_like(y_test)  # 默认所有样本预测为稳定(1)
    final_pred[non_stable_pred_idx] = second_level_pred  # 更新预测为非稳定的样本
    
    # 评估整合后的模型
    print("\n===== 层级分类最终结果 =====")
    final_accuracy = accuracy_score(y_test, final_pred) * 100
    final_conf_matrix = confusion_matrix(y_test, final_pred)
    final_report = classification_report(y_test, final_pred, 
                                        target_names=['下跌', '稳定', '上涨'])
    
    results = {
        '三分类模型': {
            'Accuracy': multi_accuracy,
            'Confusion Matrix': multi_conf_matrix,
            'Classification Report': multi_report
        },
        '层级分类': {
            'Accuracy': final_accuracy,
            'Confusion Matrix': final_conf_matrix,
            'Classification Report': final_report
        }
    }
    
    print(f"\n层级分类 准确率: {final_accuracy:.2f}%")
    print("混淆矩阵:")
    print(pd.DataFrame(
        final_conf_matrix,
        index=["实际 下跌", "实际 稳定", "实际 上涨"],
        columns=["预测 下跌", "预测 稳定", "预测 上涨"]
    ))
    print("\n分类报告:")
    print(final_report)
    
    # 收集特征重要性
    feature_importance = {}
    
    stable_importances = np.abs(stable_model.coef_[0])
    feature_importance['稳定性判断'] = pd.DataFrame({
        '特征': feature_cols,
        '重要性': stable_importances
    }).sort_values(by='重要性', ascending=False)
    
    # 多分类模型的特征重要性（取平均绝对值）
    multi_importances = np.mean(np.abs(multi_model.coef_), axis=0)
    feature_importance['三分类模型'] = pd.DataFrame({
        '特征': feature_cols,
        '重要性': multi_importances
    }).sort_values(by='重要性', ascending=False)
    
    return results, feature_importance

# 主函数
def main():
    # 配置
    target_labels = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
    
    # 加载数据
    df = load_data(files)
    
    for target_label in target_labels:
        print(f"\n处理目标: {target_label}")
        
        # 数据预处理
        X, y, feature_cols = preprocess_data(df, target_label)
        
        # 划分数据
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # 训练和评估模型
        results, feature_importance = train_evaluate_models(
            X_train, X_test, y_train, y_test, feature_cols
        )
        
        # 可视化结果

if __name__ == "__main__":
    main()