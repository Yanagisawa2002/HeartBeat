import torch
import numpy as np
import os
from typing import Tuple, List
# torch_geometric import removed
import yaml

class ECGDataAdapter:
    """ECG数据适配器，将图数据转换为传统深度学习模型格式"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
    def graph_to_sequence(self, graph_data: List[Data]) -> Tuple[np.ndarray, np.ndarray]:
        """将图数据转换为序列数据
        
        Args:
            graph_data: 图数据列表
            
        Returns:
            X: 序列数据 (num_samples, num_leads, seq_len)
            y: 标签数据 (num_samples,)
        """
        X_list = []
        y_list = []
        
        for graph in graph_data:
            # 从图节点特征重构ECG信号
            node_features = graph.x.numpy()  # (num_nodes, feature_dim)
            
            # 假设节点特征包含了12导联的信息
            # 这里需要根据实际的图构建方式来调整
            if node_features.shape[1] >= 12:
                # 如果节点特征维度>=12，取前12维作为12导联
                ecg_signal = node_features[:, :12].T  # (12, seq_len)
            else:
                # 如果特征维度<12，进行填充或重复
                ecg_signal = np.tile(node_features.T, (12, 1))[:12, :]  # (12, seq_len)
                
            X_list.append(ecg_signal)
            y_list.append(graph.y.item())
            
        # 统一序列长度
        max_len = max(x.shape[1] for x in X_list)
        X_padded = []
        
        for x in X_list:
            if x.shape[1] < max_len:
                # 零填充
                padding = np.zeros((12, max_len - x.shape[1]))
                x_padded = np.concatenate([x, padding], axis=1)
            else:
                x_padded = x[:, :max_len]
            X_padded.append(x_padded)
            
        X = np.array(X_padded)  # (num_samples, 12, seq_len)
        y = np.array(y_list)    # (num_samples,)
        
        return X, y
        
    def load_and_convert_graph_data(self) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """加载并转换图数据
        
        Returns:
            X_data: (X_train, X_val, X_test)
            y_data: (y_train, y_val, y_test)
        """
        print("Loading and converting graph data...")
        
        # 图数据路径
        graph_paths = {
            'train': self.config['data']['train_graph_path'],
            'val': self.config['data']['val_graph_path'],
            'test': self.config['data']['test_graph_path']
        }
        
        X_data = {}
        y_data = {}
        
        for split, path in graph_paths.items():
            if os.path.exists(path):
                print(f"Loading {split} graph data from {path}")
                graph_data = torch.load(path)
                X, y = self.graph_to_sequence(graph_data)
                X_data[split] = X
                y_data[split] = y
                print(f"{split} data shape: X={X.shape}, y={y.shape}")
            else:
                print(f"Graph data not found: {path}")
                return None, None
                
        return (X_data['train'], X_data['val'], X_data['test']), \
               (y_data['train'], y_data['val'], y_data['test'])
               
    def create_synthetic_ecg_data(self, num_samples: int = 1000, seq_len: int = 1000, 
                                 num_leads: int = 12) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """创建合成ECG数据用于测试
        
        Args:
            num_samples: 样本数量
            seq_len: 序列长度
            num_leads: 导联数量
            
        Returns:
            X_data: (X_train, X_val, X_test)
            y_data: (y_train, y_val, y_test)
        """
        print(f"Creating synthetic ECG data: {num_samples} samples, {seq_len} length, {num_leads} leads")
        
        # 生成合成ECG信号
        np.random.seed(42)
        
        # 创建基础ECG模式
        t = np.linspace(0, 10, seq_len)
        
        X_all = []
        y_all = []
        
        for i in range(num_samples):
            # 生成12导联ECG信号
            ecg_signal = np.zeros((num_leads, seq_len))
            
            # 基础心率和节律
            heart_rate = np.random.uniform(60, 100)  # 心率
            frequency = heart_rate / 60  # Hz
            
            for lead in range(num_leads):
                # P波、QRS波群、T波的合成
                p_wave = 0.1 * np.sin(2 * np.pi * frequency * t + np.random.uniform(0, 0.5))
                qrs_complex = 0.8 * np.sin(2 * np.pi * frequency * 3 * t + np.random.uniform(0, 0.3))
                t_wave = 0.2 * np.sin(2 * np.pi * frequency * 0.5 * t + np.random.uniform(0, 0.7))
                
                # 添加噪声
                noise = 0.05 * np.random.randn(seq_len)
                
                # 合成信号
                ecg_signal[lead] = p_wave + qrs_complex + t_wave + noise
                
                # 导联间的相关性
                if lead > 0:
                    correlation = np.random.uniform(0.3, 0.8)
                    ecg_signal[lead] = correlation * ecg_signal[0] + (1 - correlation) * ecg_signal[lead]
                    
            # 生成标签（50%正常，50%异常）
            if i < num_samples // 2:
                label = 0  # 正常
            else:
                label = 1  # 异常
                # 为异常样本添加特殊模式
                if np.random.random() < 0.5:
                    # 心律不齐
                    irregular_pattern = 0.3 * np.sin(2 * np.pi * frequency * 2.5 * t)
                    ecg_signal += irregular_pattern
                else:
                    # ST段异常
                    st_deviation = 0.2 * np.ones_like(t)
                    ecg_signal += st_deviation
                    
            X_all.append(ecg_signal)
            y_all.append(label)
            
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        
        # 划分数据集
        train_size = int(0.7 * num_samples)
        val_size = int(0.15 * num_samples)
        
        indices = np.random.permutation(num_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        X_train = X_all[train_indices]
        X_val = X_all[val_indices]
        X_test = X_all[test_indices]
        
        y_train = y_all[train_indices]
        y_val = y_all[val_indices]
        y_test = y_all[test_indices]
        
        print(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)
        
    def save_converted_data(self, X_data: Tuple[np.ndarray, ...], y_data: Tuple[np.ndarray, ...], 
                           save_path: str = "data/processed/converted_ecg_data.npz"):
        """保存转换后的数据
        
        Args:
            X_data: (X_train, X_val, X_test)
            y_data: (y_train, y_val, y_test)
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        np.savez(save_path,
                X_train=X_data[0], X_val=X_data[1], X_test=X_data[2],
                y_train=y_data[0], y_val=y_data[1], y_test=y_data[2])
                
        print(f"Converted data saved to {save_path}")
        
    def load_converted_data(self, load_path: str = "data/processed/converted_ecg_data.npz") -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """加载转换后的数据
        
        Args:
            load_path: 数据路径
            
        Returns:
            X_data: (X_train, X_val, X_test)
            y_data: (y_train, y_val, y_test)
        """
        if not os.path.exists(load_path):
            print(f"Converted data not found at {load_path}")
            return None, None
            
        data = np.load(load_path)
        
        X_data = (data['X_train'], data['X_val'], data['X_test'])
        y_data = (data['y_train'], data['y_val'], data['y_test'])
        
        print(f"Loaded converted data from {load_path}")
        print(f"Train: {X_data[0].shape}, Val: {X_data[1].shape}, Test: {X_data[2].shape}")
        
        return X_data, y_data
        
    def get_data_for_comparison(self) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """获取用于模型比较的数据
        
        Returns:
            X_data: (X_train, X_val, X_test)
            y_data: (y_train, y_val, y_test)
        """
        # 首先尝试加载转换后的数据
        converted_data_path = "data/processed/converted_ecg_data.npz"
        X_data, y_data = self.load_converted_data(converted_data_path)
        
        if X_data is not None:
            return X_data, y_data
            
        # 尝试加载单独的.npy文件
        try:
            X_train = np.load("data/processed/X_train.npy")
            X_val = np.load("data/processed/X_val.npy")
            X_test = np.load("data/processed/X_test.npy")
            y_train = np.load("data/processed/y_train.npy")
            y_val = np.load("data/processed/y_val.npy")
            y_test = np.load("data/processed/y_test.npy")
            
            print(f"Loaded processed data from .npy files")
            print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            return (X_train, X_val, X_test), (y_train, y_val, y_test)
        except Exception as e:
            print(f"Error loading .npy files: {e}")
            
        # 如果没有转换后的数据，尝试从图数据转换
        print("Trying to convert from graph data...")
        X_data, y_data = self.load_and_convert_graph_data()
        
        if X_data is not None:
            # 保存转换后的数据
            self.save_converted_data(X_data, y_data, converted_data_path)
            return X_data, y_data
            
        # 如果都没有，创建合成数据
        print("Creating synthetic data for comparison...")
        X_data, y_data = self.create_synthetic_ecg_data(num_samples=2000, seq_len=1000)
        
        # 保存合成数据
        self.save_converted_data(X_data, y_data, converted_data_path)
        
        return X_data, y_data
        
def main():
    """测试数据适配器"""
    print("Testing ECG Data Adapter...")
    
    adapter = ECGDataAdapter()
    
    # 获取比较数据
    X_data, y_data = adapter.get_data_for_comparison()
    
    if X_data is not None:
        print("\nData shapes:")
        print(f"X_train: {X_data[0].shape}")
        print(f"X_val: {X_data[1].shape}")
        print(f"X_test: {X_data[2].shape}")
        print(f"y_train: {y_data[0].shape}")
        print(f"y_val: {y_data[1].shape}")
        print(f"y_test: {y_data[2].shape}")
        
        # 检查标签分布
        print("\nLabel distribution:")
        for split, y in zip(['train', 'val', 'test'], y_data):
            unique, counts = np.unique(y, return_counts=True)
            print(f"{split}: {dict(zip(unique, counts))}")
            
        print("\nData adapter test completed successfully!")
    else:
        print("Failed to get comparison data.")
        
if __name__ == "__main__":
    main()