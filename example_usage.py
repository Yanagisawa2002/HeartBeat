#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTB心电图异常检测 - 使用示例

本脚本展示如何使用项目进行心电图异常检测的完整流程。

作者: HeartBeat项目组
日期: 2024
"""

import os
import sys
import numpy as np
import torch
from typing import List, Tuple

# 添加src目录到路径
sys.path.append('src')

from src.data_loader import PTBDataLoader

from src.train import ECGTrainer
from src.evaluate import ECGEvaluator

def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("PTB心电图异常检测 - 环境检查")
    print("=" * 60)
    
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name()}")
    
    print("\n必要目录检查:")
    directories = ['data', 'data/raw', 'data/processed', 'src', 'models', 'results', 'logs']
    for directory in directories:
        exists = os.path.exists(directory)
        status = "✓" if exists else "✗"
        print(f"{status} {directory}")
    
    print("\n配置文件检查:")
    config_exists = os.path.exists('config.yaml')
    status = "✓" if config_exists else "✗"
    print(f"{status} config.yaml")
    
    return config_exists

def create_sample_data():
    """创建示例数据用于测试"""
    print("\n创建示例数据...")
    
    # 创建模拟的心电图数据
    np.random.seed(42)
    
    # 参数设置
    num_samples = 100
    num_leads = 12
    signal_length = 5000  # 5秒，采样率1000Hz
    
    sample_data = []
    sample_labels = []
    
    for i in range(num_samples):
        # 生成基础心电图信号
        t = np.linspace(0, 5, signal_length)
        
        # 模拟心电图信号（简化版）
        ecg_signal = np.zeros((num_leads, signal_length))
        
        for lead in range(num_leads):
            # 基础正弦波 + 噪声
            base_freq = 1.2 + np.random.normal(0, 0.1)  # 心率变化
            signal = np.sin(2 * np.pi * base_freq * t)
            
            # 添加R波峰值
            r_peaks = np.where(np.diff(np.sign(np.diff(signal))) < 0)[0]
            for peak in r_peaks:
                if peak < signal_length - 10:
                    signal[peak:peak+10] += np.random.normal(2, 0.5)
            
            # 添加噪声
            noise = np.random.normal(0, 0.1, signal_length)
            signal += noise
            
            # 异常情况模拟
            if i >= num_samples // 2:  # 后一半为异常
                # 添加异常模式
                if np.random.random() > 0.5:
                    # 心律不齐
                    signal += 0.5 * np.sin(2 * np.pi * 10 * t)
                else:
                    # 幅值异常
                    signal *= np.random.uniform(1.5, 3.0)
            
            ecg_signal[lead] = signal
        
        sample_data.append(ecg_signal)
        sample_labels.append(0 if i < num_samples // 2 else 1)
    
    print(f"创建了 {len(sample_data)} 个样本")
    print(f"正常样本: {sample_labels.count(0)}")
    print(f"异常样本: {sample_labels.count(1)}")
    
    return sample_data, sample_labels

def demo_data_processing():
    """演示数据处理流程"""
    print("\n" + "=" * 60)
    print("数据处理演示")
    print("=" * 60)
    
    # 创建数据加载器
    data_loader = PTBDataLoader()
    
    # 检查是否有真实数据
    if os.path.exists(data_loader.raw_data_path) and os.listdir(data_loader.raw_data_path):
        print("发现PTB原始数据，开始处理...")
        try:
            processed_data, labels = data_loader.load_and_process_dataset()
            data_loader.save_processed_data(processed_data, labels)
            print("数据处理完成！")
        except Exception as e:
            print(f"数据处理出错: {e}")
            print("使用示例数据继续演示...")
            return create_sample_data()
    else:
        print("未找到PTB原始数据，使用示例数据演示...")
        return create_sample_data()

def demo_graph_building(sample_data: List[np.ndarray], sample_labels: List[int]):
    """演示图构建流程"""
    print("\n" + "=" * 60)
    print("图构建演示")
    print("=" * 60)
    

    print("图构建功能已移除")
    return None

def demo_model_creation():
    """演示模型创建"""
    print("\n" + "=" * 60)
    print("模型创建演示")
    print("=" * 60)
    
    
    
    return None

def demo_training_setup():
    """演示训练设置"""
    print("\n" + "=" * 60)
    print("训练设置演示")
    print("=" * 60)
    
    # 创建训练器
    trainer = ECGTrainer()
    
    print("训练配置:")
    print(f"  设备: {trainer.device}")
    print(f"  批次大小: {trainer.batch_size}")
    print(f"  学习率: {trainer.learning_rate}")
    print(f"  训练轮数: {trainer.num_epochs}")
    print(f"  早停耐心: {trainer.early_stopping_patience}")
    
    print("\n要开始完整训练，请运行:")
    print("python src/train.py")
    
    return trainer

def demo_evaluation_setup():
    """演示评估设置"""
    print("\n" + "=" * 60)
    print("评估设置演示")
    print("=" * 60)
    
    # 创建评估器
    evaluator = ECGEvaluator()
    
    print("评估配置:")
    eval_config = evaluator.eval_config
    for key, value in eval_config.items():
        print(f"  {key}: {value}")
    
    # 检查是否有训练好的模型
    model_path = os.path.join(evaluator.config['paths']['model_save_path'], 'best_model.pth')
    
    if os.path.exists(model_path):
        print(f"\n发现训练好的模型: {model_path}")
        print("要开始评估，请运行:")
        print("python src/evaluate.py")
    else:
        print("\n未找到训练好的模型")
        print("请先运行训练脚本: python src/train.py")
    
    return evaluator

def print_usage_instructions():
    """打印使用说明"""
    print("\n" + "=" * 60)
    print("使用说明")
    print("=" * 60)
    
    instructions = """
完整使用流程:

1. 数据准备:
   - 下载PTB数据库
   - 将数据文件放置在 data/raw/ 目录下
   - 数据应包含 .hea 和 .dat 文件

2. 数据预处理:
   python src/data_loader.py
   
3. 模型训练:
   python src/train.py
   
4. 模型评估:
   python src/evaluate.py
   
5. 查看结果:
   - 训练日志: logs/
   - 模型文件: models/
   - 评估结果: results/
   
6. Jupyter演示:
   jupyter notebook model_training_evaluation.ipynb

配置调整:
- 编辑 config.yaml 文件调整超参数
- 支持的模型类型: GCN, GAT, GraphSAGE, GIN, Hybrid
- 可调整图构建参数、训练参数等

注意事项:
- 确保安装了所有依赖包: pip install -r requirements.txt
- 建议使用GPU进行训练以提高速度
- 根据数据集大小调整批次大小和学习率
    """
    
    print(instructions)

def main():
    """主函数：演示完整的使用流程"""
    print("=" * 60)
    print("心电图异常检测系统 - 使用演示")
    print("=" * 60)
    
    # 检查CUDA可用性
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: {torch.cuda.get_device_name()}")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  可用GPU数量: {torch.cuda.device_count()}")
    else:
        print("⚠ CUDA不可用，将使用CPU（速度较慢）")
    
    # 环境检查
    config_exists = check_environment()
    
    if not config_exists:
        print("\n错误: 未找到配置文件 config.yaml")
        print("请确保在项目根目录下运行此脚本")
        return
    
    # 检查PTB-XL数据
    print("\n2. 检查PTB-XL数据...")
    ptbxl_path = "data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
    if os.path.exists(ptbxl_path):
        print(f"✓ 找到PTB-XL数据集: {ptbxl_path}")
        database_path = os.path.join(ptbxl_path, "ptbxl_database.csv")
        if os.path.exists(database_path):
            print(f"✓ 找到数据库文件: {database_path}")
            # 快速检查数据
            import pandas as pd
            df = pd.read_csv(database_path, nrows=5)
            print(f"  数据库列: {list(df.columns)}")
        else:
            print(f"✗ 未找到数据库文件: {database_path}")
    else:
        print(f"✗ 未找到PTB-XL数据集: {ptbxl_path}")
        print("  请确保已下载PTB-XL数据集到data/raw/目录")
        print("  使用示例数据继续演示...")
    
    try:
        # 数据处理演示
        sample_data, sample_labels = demo_data_processing()
        
        # 图构建演示
        sample_graphs = demo_graph_building(sample_data, sample_labels)
        
        # 模型创建演示
        demo_model_creation()
        
        # 训练设置演示
        trainer = demo_training_setup()
        
        # 评估设置演示
        evaluator = demo_evaluation_setup()
        
        # 使用说明
        print_usage_instructions()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("\n下一步操作：")
        print("1. 运行完整数据处理: python src/data_loader.py")
        print("2. 开始模型训练: python src/train.py")
        print("3. 评估模型性能: python src/evaluate.py")
        print("4. 查看详细教程: ecg_anomaly_detection_demo.ipynb")
        print("\n注意：")
        if torch.cuda.is_available():
            print("- 已启用CUDA加速，训练速度会更快")
        else:
            print("- 建议安装CUDA版本的PyTorch以获得更好的性能")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n演示过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()