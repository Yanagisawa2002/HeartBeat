import os
import numpy as np
import pandas as pd
import wfdb
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import yaml
import torch
from typing import Tuple, List, Dict, Optional

class PTBDataLoader:
    """PTB数据库数据加载器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备（CUDA或CPU）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"CUDA设备: {torch.cuda.get_device_name()}")
            print(f"CUDA内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.raw_data_path = self.config['data']['raw_data_path']
        self.processed_data_path = self.config['data']['processed_data_path']
        self.sampling_rate = self.config['data']['sampling_rate']
        self.signal_length = self.config['data']['signal_length']
        self.leads = self.config['data']['leads']
        
        # 创建处理后数据目录
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        # PTB-XL数据集路径
        self.ptbxl_path = os.path.join(self.raw_data_path, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1')
        self.database_path = os.path.join(self.ptbxl_path, 'ptbxl_database.csv')
    
    def load_ptb_record(self, record_path: str) -> Tuple[np.ndarray, Dict]:
        """加载单个PTB记录
        
        Args:
            record_path: 记录文件路径（不含扩展名）
            
        Returns:
            signals: 心电图信号数据 (leads, samples)
            metadata: 元数据信息
        """
        try:
            # 读取wfdb记录
            record = wfdb.rdrecord(record_path)
            
            # 获取信号数据
            signals = record.p_signal.T  # 转置为 (leads, samples)
            
            # 获取元数据
            metadata = {
                'fs': record.fs,
                'sig_len': record.sig_len,
                'sig_name': record.sig_name,
                'units': record.units,
                'comments': record.comments
            }
            
            return signals, metadata
            
        except Exception as e:
            print(f"加载记录 {record_path} 时出错: {e}")
            return None, None
    
    def preprocess_signal(self, signals: np.ndarray) -> np.ndarray:
        """预处理心电图信号
        
        Args:
            signals: 原始信号 (leads, samples)
            
        Returns:
            processed_signals: 预处理后的信号
        """
        processed_signals = []
        
        for lead_signal in signals:
            # 1. 去除基线漂移（高通滤波）
            sos_hp = signal.butter(4, 0.5, btype='high', fs=self.sampling_rate, output='sos')
            filtered_signal = signal.sosfilt(sos_hp, lead_signal)
            
            # 2. 去除高频噪声（低通滤波）- 适应100Hz采样率
            nyquist = self.sampling_rate / 2
            cutoff_freq = min(40, nyquist - 1)  # 确保截止频率小于奈奎斯特频率
            sos_lp = signal.butter(4, cutoff_freq, btype='low', fs=self.sampling_rate, output='sos')
            filtered_signal = signal.sosfilt(sos_lp, filtered_signal)
            
            # 3. 去除工频干扰（陷波滤波）- 仅在采样率足够高时应用
            if self.sampling_rate > 120:  # 确保有足够的频率范围
                sos_notch = signal.butter(4, [49, 51], btype='bandstop', fs=self.sampling_rate, output='sos')
                filtered_signal = signal.sosfilt(sos_notch, filtered_signal)
            
            # 4. 标准化
            signal_std = np.std(filtered_signal)
            if signal_std > 1e-8:  # 避免除零
                filtered_signal = (filtered_signal - np.mean(filtered_signal)) / signal_std
            else:
                filtered_signal = filtered_signal - np.mean(filtered_signal)
            
            processed_signals.append(filtered_signal)
        
        return np.array(processed_signals)
    
    def extract_features(self, signals: np.ndarray) -> np.ndarray:
        """提取心电图特征
        
        Args:
            signals: 预处理后的信号 (leads, samples)
            
        Returns:
            features: 提取的特征
        """
        features = []
        
        for lead_signal in signals:
            lead_features = []
            
            # 时域特征
            lead_features.extend([
                np.mean(lead_signal),           # 均值
                np.std(lead_signal),            # 标准差
                np.var(lead_signal),            # 方差
                np.max(lead_signal),            # 最大值
                np.min(lead_signal),            # 最小值
                np.ptp(lead_signal),            # 峰峰值
                np.mean(np.abs(lead_signal)),   # 平均绝对值
                np.sqrt(np.mean(lead_signal**2)) # RMS
            ])
            
            # 频域特征
            freqs, psd = signal.welch(lead_signal, fs=self.sampling_rate, nperseg=1024)
            
            # 不同频带的功率
            freq_bands = self.config['graph']['frequency_bands']
            for i in range(len(freq_bands)-1):
                band_mask = (freqs >= freq_bands[i]) & (freqs < freq_bands[i+1])
                band_power = np.sum(psd[band_mask])
                lead_features.append(band_power)
            
            # 主频率
            dominant_freq = freqs[np.argmax(psd)]
            lead_features.append(dominant_freq)
            
            # 频谱熵
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
            lead_features.append(spectral_entropy)
            
            features.append(lead_features)
        
        return np.array(features)
    
    def segment_signal(self, signals: np.ndarray, segment_length: int = None) -> List[np.ndarray]:
        """将长信号分割为固定长度的片段
        
        Args:
            signals: 信号数据 (leads, samples)
            segment_length: 片段长度，默认使用配置中的signal_length
            
        Returns:
            segments: 分割后的信号片段列表
        """
        if segment_length is None:
            segment_length = self.signal_length
        
        _, total_samples = signals.shape
        segments = []
        
        # 滑动窗口分割
        step_size = segment_length // 2  # 50%重叠
        
        for start in range(0, total_samples - segment_length + 1, step_size):
            end = start + segment_length
            segment = signals[:, start:end]
            segments.append(segment)
        
        return segments
    
    def load_ptbxl_database(self) -> pd.DataFrame:
        """加载PTB-XL数据库元数据
        
        Returns:
            df: 包含所有记录信息的DataFrame
        """
        if not os.path.exists(self.database_path):
            raise FileNotFoundError(f"PTB-XL数据库文件不存在: {self.database_path}")
        
        # 加载数据库
        df = pd.read_csv(self.database_path, index_col='ecg_id')
        return df
    
    def get_record_path(self, ecg_id: int, sampling_rate: int = 100) -> str:
        """获取记录文件路径
        
        Args:
            ecg_id: ECG记录ID
            sampling_rate: 采样率 (100 or 500)
            
        Returns:
            record_path: 记录文件路径（不含扩展名）
        """
        # 构建文件路径
        folder = f"{ecg_id:05d}"[:-3] + "000"  # 例如：00001 -> 00000
        filename = f"{ecg_id:05d}_lr"  # 例如：00001_lr
        
        record_path = os.path.join(
            self.ptbxl_path, 
            f"records{sampling_rate}", 
            folder, 
            filename
        )
        
        return record_path
    
    def parse_scp_codes(self, scp_codes_str: str) -> dict:
        """解析scp_codes字符串
        
        Args:
            scp_codes_str: scp_codes字符串，如"{'NORM': 100.0, 'SR': 0.0}"
            
        Returns:
            dict: 解析后的字典
        """
        try:
            import ast
            return ast.literal_eval(scp_codes_str)
        except:
            return {}
    
    def is_normal_record(self, scp_codes_str: str) -> bool:
        """判断记录是否为正常
        
        Args:
            scp_codes_str: scp_codes字符串
            
        Returns:
            bool: True表示正常，False表示异常
        """
        scp_dict = self.parse_scp_codes(scp_codes_str)
        
        # 如果包含NORM且置信度较高，认为是正常
        if 'NORM' in scp_dict and scp_dict['NORM'] >= 50.0:
            # 检查是否有其他高置信度的异常标签
            for code, confidence in scp_dict.items():
                if code != 'NORM' and code != 'SR' and confidence >= 50.0:
                    return False  # 有其他高置信度异常，认为是异常
            return True
        
        return False  # 不包含NORM或置信度低，认为是异常
    
    def load_and_process_dataset(self, max_samples: int = None) -> Tuple[List[np.ndarray], List[int]]:
        """加载并处理整个数据集
        
        Args:
            max_samples: 最大样本数量，用于测试
            
        Returns:
            processed_data: 处理后的数据列表
            labels: 标签列表 (0: 正常, 1: 异常)
        """
        processed_data = []
        labels = []
        
        # 加载PTB-XL数据库
        print("加载PTB-XL数据库...")
        df = self.load_ptbxl_database()
        
        # 定义正常和异常的标准
        print("分析scp_codes标签...")
        normal_records = []
        abnormal_records = []
        
        for idx, row in df.iterrows():
            if pd.isna(row['scp_codes']):
                continue
            
            if self.is_normal_record(row['scp_codes']):
                normal_records.append(idx)
            else:
                abnormal_records.append(idx)
        
        print(f"找到 {len(normal_records)} 个正常记录")
        print(f"找到 {len(abnormal_records)} 个异常记录")
        
        # 确保有足够的异常样本
        if len(abnormal_records) < len(normal_records) * 0.1:  # 异常样本少于10%
            print(f"异常样本过少({len(abnormal_records)})，从正常样本中随机选择一些作为异常")
            np.random.seed(42)
            additional_abnormal = min(len(normal_records) // 4, 1000)  # 最多选择1000个
            selected_indices = np.random.choice(normal_records, additional_abnormal, replace=False)
            
            # 将选中的从正常中移除，加入异常
            for idx in selected_indices:
                normal_records.remove(idx)
                abnormal_records.append(idx)
            
            print(f"调整后: {len(normal_records)} 个正常记录, {len(abnormal_records)} 个异常记录")
        
        # 限制样本数量（用于测试）
        if max_samples:
            max_normal = min(max_samples // 2, len(normal_records))
            max_abnormal = min(max_samples - max_normal, len(abnormal_records))
            normal_records = normal_records[:max_normal]
            abnormal_records = abnormal_records[:max_abnormal]
        
        # 处理正常记录
        print(f"处理 {len(normal_records)} 个正常记录...")
        for i, ecg_id in enumerate(normal_records):
            if i % 100 == 0:
                print(f"  进度: {i}/{len(normal_records)}")
            
            record_path = self.get_record_path(ecg_id)
            signals, metadata = self.load_ptb_record(record_path)
            
            if signals is not None:
                try:
                    # 预处理
                    processed_signals = self.preprocess_signal(signals)
                    
                    # 分割信号
                    segments = self.segment_signal(processed_signals)
                    
                    for segment in segments:
                        processed_data.append(segment)
                        labels.append(0)  # 正常
                except Exception as e:
                    print(f"处理记录 {ecg_id} 时出错: {e}")
                    continue
        
        # 处理异常记录
        print(f"处理 {len(abnormal_records)} 个异常记录...")
        for i, ecg_id in enumerate(abnormal_records):
            if i % 100 == 0:
                print(f"  进度: {i}/{len(abnormal_records)}")
            
            record_path = self.get_record_path(ecg_id)
            signals, metadata = self.load_ptb_record(record_path)
            
            if signals is not None:
                try:
                    # 预处理
                    processed_signals = self.preprocess_signal(signals)
                    
                    # 分割信号
                    segments = self.segment_signal(processed_signals)
                    
                    for segment in segments:
                        processed_data.append(segment)
                        labels.append(1)  # 异常
                except Exception as e:
                    print(f"处理记录 {ecg_id} 时出错: {e}")
                    continue
        
        print(f"数据处理完成！总共 {len(processed_data)} 个样本")
        print(f"正常样本: {labels.count(0)}, 异常样本: {labels.count(1)}")
        
        return processed_data, labels
    
    def save_processed_data(self, data: List[np.ndarray], labels: List[int]):
        """保存处理后的数据"""
        if len(data) == 0:
            raise ValueError("没有数据可保存")
        
        # 转换为numpy数组
        data_array = np.array(data)
        labels_array = np.array(labels)
        
        print(f"数据形状: {data_array.shape}")
        print(f"标签分布: 正常={np.sum(labels_array==0)}, 异常={np.sum(labels_array==1)}")
        
        # 划分训练、验证、测试集
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        
        try:
            # 首先划分训练和临时集
            X_train, X_temp, y_train, y_temp = train_test_split(
                data_array, labels_array, test_size=(1-train_ratio), random_state=42, stratify=labels_array
            )
            
            # 再划分验证和测试集
            val_size = val_ratio / (val_ratio + self.config['data']['test_ratio'])
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=(1-val_size), random_state=42, stratify=y_temp
            )
        except ValueError as e:
            print(f"分层采样失败，使用随机采样: {e}")
            # 首先划分训练和临时集
            X_train, X_temp, y_train, y_temp = train_test_split(
                data_array, labels_array, test_size=(1-train_ratio), random_state=42
            )
            
            # 再划分验证和测试集
            val_size = val_ratio / (val_ratio + self.config['data']['test_ratio'])
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=(1-val_size), random_state=42
            )
        
        # 保存数据为numpy格式（更适合CUDA加载）
        datasets = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        for split, (X, y) in datasets.items():
            # 保存为numpy格式
            np.save(os.path.join(self.processed_data_path, f'X_{split}.npy'), X)
            np.save(os.path.join(self.processed_data_path, f'y_{split}.npy'), y)
            print(f"保存 {split} 数据: {X.shape}")
        
        # 显示CUDA信息
        if torch.cuda.is_available():
            print(f"CUDA可用，使用GPU: {torch.cuda.get_device_name()}")
        else:
            print("CUDA不可用，使用CPU")
    
    def load_processed_data(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载处理后的数据
        
        Args:
            split: 数据集划分 ('train', 'val', 'test')
            
        Returns:
            data: 数据数组
            labels: 标签数组
        """
        X_path = os.path.join(self.processed_data_path, f'X_{split}.npy')
        y_path = os.path.join(self.processed_data_path, f'y_{split}.npy')
        
        if not os.path.exists(X_path) or not os.path.exists(y_path):
            # 尝试加载旧格式
            old_path = os.path.join(self.processed_data_path, f'{split}_data.pkl')
            if os.path.exists(old_path):
                with open(old_path, 'rb') as f:
                    data, labels = pickle.load(f)
                return np.array(data), np.array(labels)
            else:
                raise FileNotFoundError(f"处理后的数据文件不存在: {X_path} 或 {y_path}")
        
        data = np.load(X_path)
        labels = np.load(y_path)
        
        return data, labels

    def process_and_save_data(self, output_dir: str = None, max_samples: int = None):
        """处理数据并保存到文件
        
        Args:
            output_dir: 输出目录，默认为配置中的processed_data_path
            max_samples: 最大样本数量，用于测试
        """
        if output_dir is None:
            output_dir = self.processed_data_path
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载和处理数据
        processed_data, labels = self.load_and_process_dataset(max_samples=max_samples)
        
        if len(processed_data) == 0:
            raise ValueError("没有成功处理任何数据，请检查数据路径和格式")
        
        # 保存处理后的数据
        self.save_processed_data(processed_data, labels)
        print("数据处理完成！")

def main():
    """主函数：处理PTB数据"""
    print("开始处理PTB数据库...")
    
    # 创建数据加载器
    data_loader = PTBDataLoader()
    
    # 检查原始数据是否存在
    if not os.path.exists(data_loader.raw_data_path):
        print(f"警告: 原始数据路径不存在: {data_loader.raw_data_path}")
        print("请将PTB数据库文件放置在该目录下")
        return
    
    # 加载和处理数据集
    try:
        # 使用新的处理和保存方法
        data_loader.process_and_save_data(max_samples=None)  # 设置max_samples=100进行测试
        
    except Exception as e:
        print(f"数据处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()