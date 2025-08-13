#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整模型评估脚本
对所有训练了100个epoch的模型进行全面评估
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 添加src目录到路径
sys.path.append('src')
from src.comparison_models import create_comparison_model


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_test_data():
    """加载测试数据"""
    print("Loading test data...")
    
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Class distribution: {np.bincount(y_test)}")
    
    # 转换为PyTorch张量
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # 创建数据加载器
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return test_loader, X_test, y_test

def evaluate_model_comprehensive(model, test_loader, device, model_name):
    """全面评估模型性能"""
    print(f"\nEvaluating {model_name}...")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # 测量推理时间
            start_time = time.time()
            output = model(data)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # 获取预测和概率
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 异常类的概率
    
    # 计算各种指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    # 计算每个类别的指标
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # ROC曲线数据
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    
    # PR曲线数据
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    
    # 总推理时间
    total_inference_time = sum(inference_times)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'inference_time': total_inference_time,
        'predictions': all_preds,
        'probabilities': all_probs,
        'true_labels': all_labels
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC Score: {auc:.4f}")
    print(f"  Inference Time: {total_inference_time:.4f}s")
    
    return results

def load_and_evaluate_model(model_path, model_type, test_loader, device):
    """加载并评估指定模型"""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    try:
        # 创建模型
        model = create_comparison_model(model_type, input_dim=12, seq_len=500, num_classes=2)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        
        # 评估模型
        results = evaluate_model_comprehensive(model, test_loader, device, model_type)
        
        # 添加参数数量
        results['parameters'] = sum(p.numel() for p in model.parameters())
        
        return results
        
    except Exception as e:
        print(f"Error loading/evaluating {model_type}: {e}")
        return None

def create_detailed_report(all_results):
    """创建详细的评估报告"""
    print("\n" + "="*80)
    print("详细模型评估报告")
    print("="*80)
    print(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"评估模型数量: {len(all_results)}")
    
    # 按AUC分数排序
    sorted_results = sorted(all_results, key=lambda x: x['auc'], reverse=True)
    
    print("\n📊 模型性能排名 (按AUC分数):")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"\n{i}. {result['model_name']}")
        print(f"   准确率: {result['accuracy']:.4f}")
        print(f"   精确率: {result['precision']:.4f}")
        print(f"   召回率: {result['recall']:.4f}")
        print(f"   F1分数: {result['f1']:.4f}")
        print(f"   AUC分数: {result['auc']:.4f}")
        print(f"   推理时间: {result['inference_time']:.4f}秒")
        print(f"   参数数量: {result['parameters']:,}")
        
        # 每个类别的详细指标
        print(f"   正常类 - 精确率: {result['precision_per_class'][0]:.4f}, 召回率: {result['recall_per_class'][0]:.4f}, F1: {result['f1_per_class'][0]:.4f}")
        print(f"   异常类 - 精确率: {result['precision_per_class'][1]:.4f}, 召回率: {result['recall_per_class'][1]:.4f}, F1: {result['f1_per_class'][1]:.4f}")
    
    # 性能对比分析
    print("\n🔍 性能分析:")
    print("-" * 50)
    
    best_accuracy = max(all_results, key=lambda x: x['accuracy'])
    best_auc = max(all_results, key=lambda x: x['auc'])
    fastest_inference = min(all_results, key=lambda x: x['inference_time'])
    most_efficient = min(all_results, key=lambda x: x['parameters'])
    
    print(f"• 最高准确率: {best_accuracy['model_name']} ({best_accuracy['accuracy']:.4f})")
    print(f"• 最高AUC分数: {best_auc['model_name']} ({best_auc['auc']:.4f})")
    print(f"• 最快推理: {fastest_inference['model_name']} ({fastest_inference['inference_time']:.4f}秒)")
    print(f"• 最少参数: {most_efficient['model_name']} ({most_efficient['parameters']:,}个)")
    
    return sorted_results

def save_comprehensive_results(all_results):
    """保存完整的评估结果"""
    # 更新CSV文件
    csv_data = []
    for result in all_results:
        csv_data.append({
            'Model': result['model_name'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1 Score': result['f1'],
            'AUC Score': result['auc'],
            'Training Time (s)': 3000.0,  # 估计值，实际训练时间可能不同
            'Inference Time (s)': result['inference_time'],
            'Parameters': result['parameters']
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = 'results/comprehensive_model_evaluation.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 完整评估结果已保存到: {csv_path}")
    
    # 保存详细结果到pickle文件
    import pickle
    pickle_path = 'results/detailed_evaluation_results.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"💾 详细评估数据已保存到: {pickle_path}")

def create_comparison_visualizations(all_results):
    """创建对比可视化图表"""
    print("\n🎨 生成可视化图表...")
    
    # 确保结果目录存在
    os.makedirs('results/comprehensive_evaluation', exist_ok=True)
    
    # 1. 性能对比柱状图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('模型性能全面对比', fontsize=16, fontweight='bold')
    
    models = [r['model_name'] for r in all_results]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(models)]
    
    # 准确率
    accuracies = [r['accuracy'] for r in all_results]
    axes[0,0].bar(models, accuracies, color=colors)
    axes[0,0].set_title('准确率对比')
    axes[0,0].set_ylabel('准确率')
    axes[0,0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(accuracies):
        axes[0,0].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
    
    # AUC分数
    aucs = [r['auc'] for r in all_results]
    axes[0,1].bar(models, aucs, color=colors)
    axes[0,1].set_title('AUC分数对比')
    axes[0,1].set_ylabel('AUC分数')
    axes[0,1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(aucs):
        axes[0,1].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
    
    # 推理时间
    inf_times = [r['inference_time'] for r in all_results]
    axes[1,0].bar(models, inf_times, color=colors)
    axes[1,0].set_title('推理时间对比')
    axes[1,0].set_ylabel('推理时间 (秒)')
    axes[1,0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(inf_times):
        axes[1,0].text(i, v + v*0.02, f'{v:.2f}', ha='center', va='bottom')
    
    # 参数数量
    params = [r['parameters']/1e6 for r in all_results]  # 转换为百万
    axes[1,1].bar(models, params, color=colors)
    axes[1,1].set_title('模型参数数量对比')
    axes[1,1].set_ylabel('参数数量 (百万)')
    axes[1,1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(params):
        axes[1,1].text(i, v + v*0.02, f'{v:.2f}M', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/comprehensive_evaluation/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. ROC曲线对比
    plt.figure(figsize=(10, 8))
    for i, result in enumerate(all_results):
        plt.plot(result['fpr'], result['tpr'], 
                label=f"{result['model_name']} (AUC = {result['auc']:.3f})",
                color=colors[i], linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('ROC曲线对比')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('results/comprehensive_evaluation/roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 可视化图表已保存到 results/comprehensive_evaluation/ 目录")

def main():
    """主函数"""
    print("🚀 开始全面模型评估...")
    print("="*60)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载测试数据
    test_loader, X_test, y_test = load_test_data()
    
    # 定义要评估的模型
    models_to_evaluate = [
        ('results/comparison/models/cnn1d_epoch_100.pth', 'CNN1D'),
        ('results/comparison/models/lstm_epoch_100.pth', 'LSTM'),
        ('results/comparison/models/resnet1d_epoch_100.pth', 'RESNET1D'),
        ('results/comparison/models/hybrid_cnn_lstm_best.pth', 'HYBRID_CNN_LSTM'),
    
    ]
    
    all_results = []
    
    # 评估每个模型
    for model_path, model_type in models_to_evaluate:
        print(f"\n{'='*40}")
        print(f"评估模型: {model_type}")
        print(f"模型路径: {model_path}")
        
        result = load_and_evaluate_model(model_path, model_type, test_loader, device)
        if result:
            all_results.append(result)
        else:
            print(f"❌ 跳过模型 {model_type}")
    
    if not all_results:
        print("❌ 没有成功评估任何模型")
        return
    
    # 创建详细报告
    sorted_results = create_detailed_report(all_results)
    
    # 保存结果
    save_comprehensive_results(all_results)
    
    # 创建可视化
    create_comparison_visualizations(all_results)
    
    print("\n🎉 全面模型评估完成!")
    print("\n📁 生成的文件:")
    print("  • results/comprehensive_model_evaluation.csv - CSV格式结果")
    print("  • results/detailed_evaluation_results.pkl - 详细评估数据")
    print("  • results/comprehensive_evaluation/performance_comparison.png - 性能对比图")
    print("  • results/comprehensive_evaluation/roc_comparison.png - ROC曲线对比")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()