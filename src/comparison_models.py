import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import math

class CNN1DModel(nn.Module):
    """1D CNN模型用于心电图分类"""
    
    def __init__(self, input_length: int = 1000, num_classes: int = 2, 
                 num_filters: int = 64, dropout: float = 0.2):
        super(CNN1DModel, self).__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        
        # 卷积层
        self.conv1 = nn.Conv1d(12, num_filters, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(num_filters)
        
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_filters * 4)
        
        self.conv4 = nn.Conv1d(num_filters * 4, num_filters * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(num_filters * 8)
        
        # 池化层
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc1 = nn.Linear(num_filters * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, 12, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # 全局平均池化
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class LSTMModel(nn.Module):
    """LSTM模型用于心电图分类"""
    
    def __init__(self, input_size: int = 12, hidden_size: int = 128, 
                 num_layers: int = 2, num_classes: int = 2, dropout: float = 0.2,
                 bidirectional: bool = True):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 计算LSTM输出维度
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 全连接层
        self.fc1 = nn.Linear(lstm_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 应用层归一化
        lstm_out = self.layer_norm(lstm_out)
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 全局平均池化
        pooled = torch.mean(attn_out, dim=1)
        
        # 全连接层
        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    """Transformer模型用于心电图分类"""
    
    def __init__(self, input_size: int = 12, d_model: int = 256, nhead: int = 8,
                 num_encoder_layers: int = 6, dim_feedforward: int = 1024,
                 num_classes: int = 2, dropout: float = 0.1, max_len: int = 5000):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_size = input_size
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # 输入投影
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # 位置编码
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)
        
        # 分类
        x = self.classifier(x)
        
        return x

class ResNet1DBlock(nn.Module):
    """1D ResNet块"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResNet1DBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet1DModel(nn.Module):
    """1D ResNet模型用于心电图分类"""
    
    def __init__(self, num_classes: int = 2, num_filters: int = 64):
        super(ResNet1DModel, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv1d(12, num_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet块
        self.layer1 = self._make_layer(num_filters, num_filters, 2, stride=1)
        self.layer2 = self._make_layer(num_filters, num_filters * 2, 2, stride=2)
        self.layer3 = self._make_layer(num_filters * 2, num_filters * 4, 2, stride=2)
        self.layer4 = self._make_layer(num_filters * 4, num_filters * 8, 2, stride=2)
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters * 8, num_classes)
        
    def _make_layer(self, in_channels: int, out_channels: int, 
                   num_blocks: int, stride: int):
        layers = []
        layers.append(ResNet1DBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResNet1DBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        # x shape: (batch_size, 12, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class HybridCNNLSTMModel(nn.Module):
    """CNN-LSTM混合模型"""
    
    def __init__(self, num_classes: int = 2, cnn_filters: int = 64, 
                 lstm_hidden: int = 128, dropout: float = 0.2):
        super(HybridCNNLSTMModel, self).__init__()
        
        # CNN特征提取器
        self.cnn = nn.Sequential(
            nn.Conv1d(12, cnn_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(cnn_filters * 2, cnn_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters * 4),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM序列建模
        self.lstm = nn.LSTM(
            input_size=cnn_filters * 4,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 12, seq_len)
        
        # CNN特征提取
        cnn_out = self.cnn(x)  # (batch_size, cnn_filters*4, reduced_seq_len)
        
        # 转换为LSTM输入格式
        cnn_out = cnn_out.transpose(1, 2)  # (batch_size, reduced_seq_len, cnn_filters*4)
        
        # LSTM序列建模
        lstm_out, _ = self.lstm(cnn_out)
        
        # 全局平均池化
        pooled = torch.mean(lstm_out, dim=1)
        
        # 分类
        output = self.classifier(pooled)
        
        return output

def create_comparison_model(model_name: str, input_dim: int, seq_len: int, 
                          num_classes: int = 2, device: str = 'cpu', **kwargs):
    """
    创建比较模型的工厂函数
    
    Args:
        model_name: 模型名称 ('cnn1d', 'lstm', 'transformer', 'resnet1d', 'hybrid_cnn_lstm')
        input_dim: 输入维度（导联数）
        seq_len: 序列长度
        num_classes: 分类数量
        device: 设备
        **kwargs: 其他参数
        
    Returns:
        model: 对应的模型实例
    """
    
    # 根据不同模型映射参数
    if model_name.upper() == 'CNN1D':
        model_kwargs = {
            'input_length': seq_len,
            'num_classes': num_classes,
            **kwargs
        }
        return CNN1DModel(**model_kwargs)
    elif model_name.upper() == 'LSTM':
        model_kwargs = {
            'input_size': input_dim,
            'num_classes': num_classes,
            **kwargs
        }
        return LSTMModel(**model_kwargs)
    elif model_name.upper() == 'TRANSFORMER':
        model_kwargs = {
            'input_size': input_dim,
            'num_classes': num_classes,
            'max_len': seq_len,
            **kwargs
        }
        return TransformerModel(**model_kwargs)
    elif model_name.upper() == 'RESNET1D':
        model_kwargs = {
            'num_classes': num_classes,
            **kwargs
        }
        return ResNet1DModel(**model_kwargs)
    elif model_name.upper() == 'HYBRID_CNN_LSTM':
        model_kwargs = {
            'num_classes': num_classes,
            **kwargs
        }
        return HybridCNNLSTMModel(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

if __name__ == "__main__":
    # 测试所有模型
    batch_size = 4
    seq_len = 1000
    input_size = 12
    
    # 创建测试数据
    test_data = torch.randn(batch_size, input_size, seq_len)  # CNN格式
    test_data_lstm = torch.randn(batch_size, seq_len, input_size)  # LSTM格式
    
    models = {
        'CNN1D': create_comparison_model('CNN1D', input_dim=input_size, seq_len=seq_len),
        'LSTM': create_comparison_model('LSTM', input_dim=input_size, seq_len=seq_len),
        'Transformer': create_comparison_model('Transformer', input_dim=input_size, seq_len=seq_len),
        'ResNet1D': create_comparison_model('ResNet1D', input_dim=input_size, seq_len=seq_len),
        'Hybrid_CNN_LSTM': create_comparison_model('Hybrid_CNN_LSTM', input_dim=input_size, seq_len=seq_len)
    }
    
    for name, model in models.items():
        print(f"\n测试 {name} 模型:")
        model.eval()
        
        try:
            with torch.no_grad():
                if name in ['LSTM', 'Transformer']:
                    output = model(test_data_lstm)
                else:
                    output = model(test_data)
                    
            print(f"输出形状: {output.shape}")
            print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
            
        except Exception as e:
            print(f"错误: {e}")
    
    print("\n所有模型测试完成！")