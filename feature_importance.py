import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import time

# 导入transformer3.py中的相关模型和数据加载函数
from transformer3 import (
    TransformerModel, 
    HierarchicalTransformer,
    preprocess_and_load_to_gpu,
    TimeSeriesSequenceDatasetGPU,
    device,
    files,
    base_path
)

# 检测是否在Linux环境下
import platform
if platform.system() == "Linux":
    # Linux环境使用系统默认字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'Arial Unicode MS', 'sans-serif']
else:
    # Windows环境使用SimHei
    plt.rcParams['font.sans-serif'] = ['SimHei'] 

plt.rcParams['axes.unicode_minus'] = False

# 添加缺失的数据加载函数
def load_data(files_list):
    """加载数据函数"""
    print("开始加载数据文件...")
    
    # 合并所有文件
    df_list = []
    for file in files_list:
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

def get_feature_importance(model, test_loader, feature_names, batch_size=64, device=device, hierarchical=False):
    """计算特征重要性（置换重要性法）"""
    print("计算特征重要性...")
    
    # 收集所有测试数据
    all_X = []
    all_y = []
    
    for batch_X, batch_y in tqdm(test_loader, desc="收集测试数据"):
        all_X.append(batch_X.cpu())
        all_y.append(batch_y.cpu())
    
    X_test = torch.cat(all_X, dim=0)
    y_test = torch.cat(all_y, dim=0)
    
    # 获取原始预测结果
    if hierarchical:
        # 对分层模型进行评估
        model.stability_model.eval()
        model.direction_model.eval()
        
        all_preds = []
        print("计算原始预测...")
        
        for i in tqdm(range(0, len(X_test), batch_size)):
            end = min(i + batch_size, len(X_test))
            batch = X_test[i:end].to(device)
            with torch.no_grad():
                preds = model.predict(batch)
                all_preds.extend(preds.cpu().numpy())
    else:
        # 对标准模型进行评估
        model.eval()
        all_preds = []
        print("计算原始预测...")
        
        for i in tqdm(range(0, len(X_test), batch_size)):
            end = min(i + batch_size, len(X_test))
            batch = X_test[i:end].to(device)
            with torch.no_grad():
                outputs = model(batch)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())

def get_feature_importance(model, test_loader, feature_names, batch_size=64, device=device, hierarchical=False):
    """计算特征重要性（置换重要性法）
    
    Args:
        model: 训练好的模型（TransformerModel或HierarchicalTransformer）
        test_loader: 测试数据加载器
        feature_names: 特征名称列表
        batch_size: 批处理大小
        device: 计算设备
        hierarchical: 是否为分层模型
    
    Returns:
        特征重要性字典
    """
    print("计算特征重要性...")
    
    # 收集所有测试数据
    all_X = []
    all_y = []
    
    for batch_X, batch_y in tqdm(test_loader, desc="收集测试数据"):
        all_X.append(batch_X.cpu())
        all_y.append(batch_y.cpu())
    
    X_test = torch.cat(all_X, dim=0)
    y_test = torch.cat(all_y, dim=0)
    
    # 获取原始预测结果
    if hierarchical:
        # 对分层模型进行评估
        model.stability_model.eval()
        model.direction_model.eval()
        
        all_preds = []
        print("计算原始预测...")
        
        for i in tqdm(range(0, len(X_test), batch_size)):
            end = min(i + batch_size, len(X_test))
            batch = X_test[i:end].to(device)
            with torch.no_grad():
                preds = model.predict(batch)
                all_preds.extend(preds.cpu().numpy())
    else:
        # 对标准模型进行评估
        model.eval()
        all_preds = []
        print("计算原始预测...")
        
        for i in tqdm(range(0, len(X_test), batch_size)):
            end = min(i + batch_size, len(X_test))
            batch = X_test[i:end].to(device)
            with torch.no_grad():
                outputs = model(batch)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
    
    # 计算原始准确率
    y_test_np = y_test.numpy()
    base_accuracy = accuracy_score(y_test_np, all_preds)
    print(f"原始准确率: {base_accuracy:.4f}")
    
    # 计算每个特征的重要性
    importances = []
    
    for i, feature_name in enumerate(tqdm(feature_names, desc="计算特征重要性")):
        # 复制测试数据并打乱第i个特征
        X_test_permuted = X_test.clone()
        perm_idx = torch.randperm(len(X_test))
        X_test_permuted[:, :, i] = X_test_permuted[perm_idx, :, i]
        
        # 使用打乱特征的数据进行预测
        permuted_preds = []
        
        if hierarchical:
            for j in range(0, len(X_test_permuted), batch_size):
                end = min(j + batch_size, len(X_test_permuted))
                batch = X_test_permuted[j:end].to(device)
                with torch.no_grad():
                    preds = model.predict(batch)
                    permuted_preds.extend(preds.cpu().numpy())
        else:
            for j in range(0, len(X_test_permuted), batch_size):
                end = min(j + batch_size, len(X_test_permuted))
                batch = X_test_permuted[j:end].to(device)
                with torch.no_grad():
                    outputs = model(batch)
                    _, predicted = torch.max(outputs, 1)
                    permuted_preds.extend(predicted.cpu().numpy())
        
        # 计算打乱后的准确率
        permuted_accuracy = accuracy_score(y_test_np, permuted_preds)
        
        # 特征重要性 = 原始准确率 - 打乱后的准确率
        importance = base_accuracy - permuted_accuracy
        importances.append((feature_name, importance))
        
        print(f"特征 {feature_name}: 重要性 {importance:.6f}")
    
    # 按重要性排序
    importances.sort(key=lambda x: x[1], reverse=True)
    
    return importances, base_accuracy

def visualize_feature_importance(importances, target_label, model_type="standard", top_n=20):
    """可视化特征重要性
    
    Args:
        importances: 特征重要性列表，格式为[(feature_name, importance), ...]
        target_label: 目标标签
        model_type: 模型类型，'standard'或'hierarchical'
        top_n: 显示前n个重要特征
    """
    # 创建保存目录
    save_dir = "./feature_importance"
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取前N个特征
    if len(importances) > top_n:
        top_features = importances[:top_n]
    else:
        top_features = importances
    
    # 转换为DataFrame
    df = pd.DataFrame(top_features, columns=['特征', '重要性'])
    
    # 绘图
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='重要性', y='特征', data=df)
    plt.title(f'{model_type.capitalize()}模型 - {target_label}的特征重要性（前{len(top_features)}名）')
    plt.xlabel('重要性（原始准确率-打乱后准确率）')
    plt.ylabel('特征名称')
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(f"{save_dir}/{model_type}_feature_importance_{target_label}.png", dpi=300, bbox_inches='tight')
    print(f"特征重要性图已保存至: {save_dir}/{model_type}_feature_importance_{target_label}.png")
    
    # 保存CSV数据
    df_full = pd.DataFrame(importances, columns=['特征', '重要性'])
    df_full.to_csv(f"{save_dir}/{model_type}_feature_importance_{target_label}.csv", index=False, encoding='utf-8-sig')
    print(f"特征重要性数据已保存至: {save_dir}/{model_type}_feature_importance_{target_label}.csv")

def main():
    parser = argparse.ArgumentParser(description='计算模型的特征重要性')
    parser.add_argument('--model_type', type=str, default='all', 
                       choices=['standard', 'hierarchical', 'both', 'all'],
                       help='要分析的模型类型')
    parser.add_argument('--target_labels', type=str, default='all',
                       help='目标标签列表，用逗号分隔，例如 label_5,label_10，或者使用"all"分析所有标签')
    parser.add_argument('--top_n', type=int, default=20,
                       help='显示前n个重要特征')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='评估时的批大小')
    parser.add_argument('--skip_vis', action='store_true',
                       help='跳过可视化，只生成CSV文件')
    
    args = parser.parse_args()
    
    # 获取所有要分析的标签
    from transformer3 import target_labels as all_target_labels
    if args.target_labels.lower() == 'all':
        target_labels_to_analyze = all_target_labels
    else:
        target_labels_to_analyze = [label.strip() for label in args.target_labels.split(',')]
    
    # 获取所有要分析的模型类型
    if args.model_type.lower() == 'all':
        model_types_to_analyze = ['standard', 'hierarchical']
    elif args.model_type.lower() == 'both':
        model_types_to_analyze = ['standard', 'hierarchical']
    else:
        model_types_to_analyze = [args.model_type]
    
    print(f"将分析以下标签: {target_labels_to_analyze}")
    print(f"将分析以下模型类型: {model_types_to_analyze}")
    
    # 准备数据 - 只需加载一次
    from transformer3 import files, base_path
    df = load_data(files)
    
    # 获取特征列 - 所有标签共用
    feature_cols = [col for col in df.columns if col not in ['date', 'time'] 
                   and not col.startswith('label_')]
    print(f"有效特征数量: {len(feature_cols)}")
    
    # 创建结果目录
    results_dir = "./feature_importance_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 模型参数
    d_model = 128
    nhead = 8
    num_layers = 3
    dim_feedforward = 256
    num_classes = 3
    dropout = 0.2
    
    # 存储所有结果的列表
    all_results = []
    
    # 遍历每个标签
    for target_label in target_labels_to_analyze:
        print(f"\n{'='*50}")
        print(f"开始分析标签: {target_label}")
        print(f"{'='*50}")
        
        # 为当前标签加载数据
        train_loader, val_loader, test_loader, input_dim = preprocess_and_load_to_gpu(
            df, feature_cols, target_label, seq_length=60
        )
        
        # 遍历每种模型类型
        for model_type in model_types_to_analyze:
            print(f"\n{'-'*40}")
            print(f"分析 {model_type} 模型 - {target_label}")
            print(f"{'-'*40}")
            
            # 标准模型分析
            if model_type == 'standard':
                standard_model_path = f'./models/standard_model_{target_label}.pth'
                if os.path.exists(standard_model_path):
                    print(f"加载标准模型: {standard_model_path}")
                    standard_model = TransformerModel(
                        input_dim=input_dim,
                        d_model=d_model,
                        nhead=nhead,
                        num_layers=num_layers,
                        dim_feedforward=dim_feedforward,
                        num_classes=num_classes,
                        dropout=dropout
                    ).to(device)
                    standard_model.load_state_dict(torch.load(standard_model_path))
                    
                    # 计算特征重要性
                    std_importances, std_acc = get_feature_importance(
                        standard_model, 
                        test_loader, 
                        feature_cols, 
                        batch_size=args.batch_size, 
                        hierarchical=False
                    )
                    
                    # 记录结果
                    all_results.append({
                        'model_type': 'standard',
                        'target_label': target_label,
                        'importances': std_importances,
                        'accuracy': std_acc
                    })
                    
                    # 可视化特征重要性
                    if not args.skip_vis:
                        try:
                            visualize_feature_importance(
                                std_importances, 
                                target_label, 
                                "standard", 
                                args.top_n
                            )
                        except Exception as e:
                            print(f"可视化出错，但继续分析: {e}")
                    
                    # 保存CSV结果
                    result_path = os.path.join(results_dir, f"standard_{target_label}_importance.csv")
                    pd.DataFrame(std_importances, columns=['Feature', 'Importance']).to_csv(
                        result_path, index=False
                    )
                    print(f"已保存结果到: {result_path}")
                else:
                    print(f"找不到标准模型: {standard_model_path}")
            
            # 分层模型分析
            elif model_type == 'hierarchical':
                stability_model_path = f'./models/stability_model_{target_label}.pth'
                direction_model_path = f'./models/direction_model_{target_label}.pth'
                
                if os.path.exists(stability_model_path) and os.path.exists(direction_model_path):
                    print(f"加载分层模型")
                    # 初始化分层模型
                    hierarchical_model = HierarchicalTransformer(
                        input_dim=input_dim,
                        d_model=d_model,
                        nhead=nhead,
                        num_layers=num_layers,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout
                    )
                    
                    # 加载模型权重
                    hierarchical_model.load_models()
                    
                    # 计算特征重要性
                    hier_importances, hier_acc = get_feature_importance(
                        hierarchical_model, 
                        test_loader, 
                        feature_cols, 
                        batch_size=args.batch_size, 
                        hierarchical=True
                    )
                    
                    # 记录结果
                    all_results.append({
                        'model_type': 'hierarchical',
                        'target_label': target_label,
                        'importances': hier_importances,
                        'accuracy': hier_acc
                    })
                    
                    # 可视化特征重要性
                    if not args.skip_vis:
                        try:
                            visualize_feature_importance(
                                hier_importances, 
                                target_label, 
                                "hierarchical", 
                                args.top_n
                            )
                        except Exception as e:
                            print(f"可视化出错，但继续分析: {e}")
                    
                    # 保存CSV结果
                    result_path = os.path.join(results_dir, f"hierarchical_{target_label}_importance.csv")
                    pd.DataFrame(hier_importances, columns=['Feature', 'Importance']).to_csv(
                        result_path, index=False
                    )
                    print(f"已保存结果到: {result_path}")
                else:
                    print(f"找不到完整的分层模型")
    
    # 生成汇总报告
    print("\n生成汇总报告...")
    summary_df = []
    
    for result in all_results:
        model_type = result['model_type']
        target_label = result['target_label']
        accuracy = result['accuracy']
        
        # 获取前N个特征
        top_features = result['importances'][:args.top_n]
        
        for i, (feature, importance) in enumerate(top_features):
            summary_df.append({
                'Model': model_type,
                'Label': target_label,
                'Rank': i+1,
                'Feature': feature,
                'Importance': importance,
                'Accuracy': accuracy
            })
    
    # 保存汇总报告
    if summary_df:
        summary_path = os.path.join(results_dir, "feature_importance_summary.csv")
        pd.DataFrame(summary_df).to_csv(summary_path, index=False)
        print(f"汇总报告已保存到: {summary_path}")
    
    print("\n特征重要性分析全部完成!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"总耗时: {elapsed_time:.2f}秒")

"""
python feature_importance.py --model_type all --target_labels all --skip_vis
"""