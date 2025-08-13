#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型比较运行脚本
用于比较不同深度学习模型在心电图分类任务上的性能
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_adapter import ECGDataAdapter
from src.comparison_models import create_comparison_model
from src.model_comparison import ModelComparison
from src.data_loader import PTBDataLoader


def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_device() -> torch.device:
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device



def create_dataloaders(X_data, y_data, batch_size=32, model_type='lstm'):
    """创建数据加载器"""
    X_train, X_val, X_test = X_data
    y_train, y_val, y_test = y_data
    
    # 转换为张量
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    
    # 根据模型类型调整维度顺序
    if model_type.lower() in ['cnn1d', 'resnet1d']:
        # CNN模型期望 (batch, channels, length) = (batch, leads, time_steps)
        pass  # 保持原始维度顺序
    else:
        # LSTM/Transformer模型期望 (batch, seq_len, features) = (batch, time_steps, leads)
        X_train = X_train.transpose(1, 2)
        X_val = X_val.transpose(1, 2)
        X_test = X_test.transpose(1, 2)
    
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    # 创建数据集
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device, epochs=100, learning_rate=0.001, model_name='model', save_dir='models'):
    """训练模型"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    start_time = time.time()
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # 每20个epoch保存模型
        if (epoch + 1) % 20 == 0:
            model_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc
            }, model_path)
            print(f'Model saved: {model_path}')
    
    training_time = time.time() - start_time
    return model, training_time

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    inference_time = time.time() - start_time
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # 计算AUC
    if len(np.unique(all_labels)) == 2:
        auc = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1])
    else:
        auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'inference_time': inference_time
    }

def run_custom_comparison(X_data, y_data, model_names, device, results_dir, epochs=20, batch_size=32):
    """运行自定义模型比较"""
    results = {}
    
    # 获取数据维度信息
    input_dim = X_data[0].shape[1]  # 导联数
    seq_len = X_data[0].shape[2]    # 序列长度
    
    print(f"Input dimensions: {input_dim} leads, {seq_len} time steps")
    
    for model_name in model_names:
        try:
            print(f"\n{'='*50}")
            print(f"Training {model_name.upper()} Model")
            print(f"{'='*50}")
            
            # 为每个模型创建对应的数据加载器
            train_loader, val_loader, test_loader = create_dataloaders(
                X_data, y_data, batch_size, model_type=model_name
            )
            
            # 创建模型
            model = create_comparison_model(
                model_name=model_name,
                input_dim=input_dim,
                seq_len=seq_len,
                num_classes=2,
                device=device
            )
            
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # 训练模型
            trained_model, training_time = train_model(
                model, train_loader, val_loader, device, epochs=epochs,
                model_name=model_name, save_dir=os.path.join(results_dir, 'models')
            )
            
            # 评估模型
            eval_results = evaluate_model(trained_model, test_loader, device)
            
            # 保存结果
            results[model_name.upper()] = {
                **eval_results,
                'training_time': training_time,
                'num_parameters': sum(p.numel() for p in model.parameters())
            }
            
            print(f"{model_name.upper()} Results:")
            print(f"  Accuracy: {eval_results['accuracy']:.4f}")
            print(f"  F1 Score: {eval_results['f1']:.4f}")
            print(f"  AUC Score: {eval_results['auc']:.4f}")
            print(f"  Training Time: {training_time:.2f}s")
            print(f"  Inference Time: {eval_results['inference_time']:.2f}s")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

def generate_comparison_report(results, results_dir):
    """生成比较报告"""
    if not results:
        print("No results to compare.")
        return
    
    # 创建结果DataFrame
    comparison_data = []
    
    for model_name, metrics in results.items():
        row = {
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1 Score': metrics.get('f1', 0),
            'AUC Score': metrics.get('auc', 0),
            'Training Time (s)': metrics.get('training_time', 0),
            'Inference Time (s)': metrics.get('inference_time', 0),
            'Parameters': metrics.get('num_parameters', 0)
        }
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # 保存结果表格
    df.to_csv(os.path.join(results_dir, 'model_comparison_results.csv'), index=False)
    
    # 打印结果表格
    print("\nModel Comparison Results:")
    print(df.to_string(index=False, float_format='%.4f'))
    
    # 找出最佳模型
    if len(df) > 0:
        best_model_f1 = df.loc[df['F1 Score'].idxmax(), 'Model']
        best_model_auc = df.loc[df['AUC Score'].idxmax(), 'Model']
        
        print(f"\nBest Model (F1 Score): {best_model_f1} ({df['F1 Score'].max():.4f})")
        print(f"Best Model (AUC Score): {best_model_auc} ({df['AUC Score'].max():.4f})")

def plot_comparison_results(results, results_dir):
    """绘制比较结果"""
    if not results:
        return
    
    # 准备数据
    models = list(results.keys())
    metrics = ['accuracy', 'f1', 'auc']
    metric_names = ['Accuracy', 'F1 Score', 'AUC Score']
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ECG Classification Model Comparison Results', fontsize=16, fontweight='bold')
    
    # 性能指标比较
    ax1 = axes[0, 0]
    x = np.arange(len(models))
    width = 0.25
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results[model].get(metric, 0) for model in models]
        ax1.bar(x + i*width, values, width, label=name, alpha=0.8)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 训练时间比较
    ax2 = axes[0, 1]
    training_times = [results[model].get('training_time', 0) for model in models]
    bars = ax2.bar(models, training_times, alpha=0.7, color='skyblue')
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, time_val in zip(bars, training_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    # 参数数量比较
    ax3 = axes[1, 0]
    param_counts = [results[model].get('num_parameters', 0) for model in models]
    bars = ax3.bar(models, param_counts, alpha=0.7, color='lightcoral')
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Number of Parameters')
    ax3.set_title('Model Complexity Comparison')
    ax3.tick_params(axis='x', rotation=45)
    
    # 雷达图
    ax4 = axes[1, 1]
    angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    for model in models:
        values = [results[model].get(metric, 0) for metric in metrics]
        values += values[:1]  # 闭合图形
        ax4.plot(angles, values, 'o-', linewidth=2, label=model)
        ax4.fill(angles, values, alpha=0.25)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metric_names)
    ax4.set_ylim(0, 1)
    ax4.set_title('Performance Radar Chart')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {os.path.join(results_dir, 'model_comparison_plots.png')}")

def main():
    parser = argparse.ArgumentParser(description='ECG Model Comparison')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--results_dir', type=str, default='results/comparison', help='Results directory')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 准备数据
    print("Preparing data...")
    print("Using real ECG data")
    adapter = ECGDataAdapter()
    X_data, y_data = adapter.get_data_for_comparison()
    
    # 定义要比较的模型 (只训练剩余的模型)
    model_names = ['resnet1d', 'hybrid_cnn_lstm']
    
    # 运行比较
    results = run_custom_comparison(
        X_data, y_data, model_names, device, args.results_dir, 
        epochs=args.epochs, batch_size=args.batch_size
    )

    
    # 生成比较报告和可视化
    generate_comparison_report(results, args.results_dir)
    plot_comparison_results(results, args.results_dir)
    
    print(f"\nComparison completed! Results saved to {args.results_dir}")
    
if __name__ == '__main__':
    main()