import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
# torch_geometric import removed
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import yaml
import pickle
import time
from typing import List, Tuple, Dict

from data_loader import PTBDataLoader


class ECGTrainer:
    """心电图异常检测训练器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 设置随机种子
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        
        # 创建必要的目录
        os.makedirs(self.config['paths']['model_save_path'], exist_ok=True)
        os.makedirs(self.config['paths']['results_path'], exist_ok=True)
        os.makedirs(self.config['paths']['logs_path'], exist_ok=True)
        
        # 初始化组件
        self.data_loader = PTBDataLoader(config_path)
        
        
        # 训练配置
        self.training_config = self.config['training']
        self.batch_size = self.training_config['batch_size']
        self.learning_rate = self.training_config['learning_rate']
        self.num_epochs = self.training_config['num_epochs']
        self.early_stopping_patience = self.training_config['early_stopping_patience']
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir=self.config['paths']['logs_path'])
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_auc': []
        }
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """准备训练数据
        
        Returns:
            train_loader, val_loader, test_loader: 数据加载器
        """
        print("准备训练数据...")
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            print(f"使用GPU加速: {torch.cuda.get_device_name()}")
        else:
            print("使用CPU计算")
        
        # 加载处理后的数据
        try:
            train_data, train_labels = self.data_loader.load_processed_data('train')
            val_data, val_labels = self.data_loader.load_processed_data('val')
            test_data, test_labels = self.data_loader.load_processed_data('test')
            print("✓ 成功加载处理后的数据")
        except FileNotFoundError:
            print("未找到处理后的数据，开始处理原始数据...")
            # 如果没有处理后的数据，先处理原始数据
            try:
                self.data_loader.process_and_save_data()
                
                # 重新加载
                train_data, train_labels = self.data_loader.load_processed_data('train')
                val_data, val_labels = self.data_loader.load_processed_data('val')
                test_data, test_labels = self.data_loader.load_processed_data('test')
                print("✓ 数据处理和加载完成")
            except Exception as e:
                print(f"✗ 数据处理失败: {e}")
                raise
        
        print(f"训练集: {train_data.shape} 样本")
        print(f"验证集: {val_data.shape} 样本")
        print(f"测试集: {test_data.shape} 样本")
        print(f"标签分布 - 训练集: 正常={np.sum(train_labels==0)}, 异常={np.sum(train_labels==1)}")
        print(f"标签分布 - 验证集: 正常={np.sum(val_labels==0)}, 异常={np.sum(val_labels==1)}")
        print(f"标签分布 - 测试集: 正常={np.sum(test_labels==0)}, 异常={np.sum(test_labels==1)}")
        

        print("图数据构建功能已移除")
        
        print(f"训练图: {len(train_graphs)} 个")
        print(f"验证图: {len(val_graphs)} 个")
        print(f"测试图: {len(test_graphs)} 个")
        
        # 将图数据移动到指定设备
        print(f"将数据移动到设备: {self.device}")
        for graph in train_graphs + val_graphs + test_graphs:
            graph.x = graph.x.to(self.device)
            graph.edge_index = graph.edge_index.to(self.device)
            graph.y = graph.y.to(self.device)
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                graph.edge_attr = graph.edge_attr.to(self.device)
        
        # 创建数据加载器
        train_loader = DataLoader(train_graphs, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def create_model(self, input_dim: int) -> nn.Module:
        """创建模型
        
        Args:
            input_dim: 输入特征维度
            
        Returns:
            model: 深度学习模型
        """
        model = self.model_factory.create_model(input_dim=input_dim)
        model = model.to(self.device)
        
        print(f"模型类型: {self.model_factory.model_config['name']}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model
    
    def create_optimizer_and_scheduler(self, model: nn.Module) -> Tuple[optim.Optimizer, object]:
        """创建优化器和学习率调度器
        
        Args:
            model: 模型
            
        Returns:
            optimizer, scheduler: 优化器和调度器
        """
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.training_config['weight_decay']
        )
        
        scheduler_type = self.training_config.get('scheduler', 'StepLR')
        
        if scheduler_type == 'StepLR':
            scheduler = StepLR(
                optimizer,
                step_size=self.training_config['scheduler_step_size'],
                gamma=self.training_config['scheduler_gamma']
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None
        
        return optimizer, scheduler
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """训练一个epoch
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            optimizer: 优化器
            criterion: 损失函数
            
        Returns:
            avg_loss, accuracy: 平均损失和准确率
        """
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc="训练")
        for batch in pbar:
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, model: nn.Module, val_loader: DataLoader, 
                criterion: nn.Module) -> Dict[str, float]:
        """验证模型
        
        Args:
            model: 模型
            val_loader: 验证数据加载器
            criterion: 损失函数
            
        Returns:
            metrics: 验证指标字典
        """
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="验证")
            for batch in pbar:
                batch = batch.to(self.device)
                
                outputs = model(batch)
                loss = criterion(outputs, batch.y)
                
                total_loss += loss.item()
                
                # 预测和概率
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # 异常类的概率
        
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # AUC（如果有两个类别）
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        return metrics
    
    def save_model(self, model: nn.Module, epoch: int, metrics: Dict[str, float], 
                  is_best: bool = False):
        """保存模型
        
        Args:
            model: 模型
            epoch: 当前epoch
            metrics: 验证指标
            is_best: 是否为最佳模型
        """
        model_info = self.model_factory.get_model_info()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'model_config': model_info,
            'training_config': self.training_config
        }
        
        # 保存最新模型
        latest_path = os.path.join(self.config['paths']['model_save_path'], 'latest_model.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config['paths']['model_save_path'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型 (F1: {metrics['f1']:.4f})")
    
    def train(self):
        """完整的训练流程"""
        print("开始训练心电图异常检测模型...")
        
        # 准备数据
        train_loader, val_loader, test_loader = self.prepare_data()
        
        # 获取输入维度
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch.x.size(1)
        print(f"输入特征维度: {input_dim}")
        
        # 创建模型
        model = self.create_model(input_dim)
        
        # 创建优化器和调度器
        optimizer, scheduler = self.create_optimizer_and_scheduler(model)
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 早停机制
        best_f1 = 0.0
        patience_counter = 0
        
        print(f"\n开始训练 {self.num_epochs} 个epochs...")
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # 验证
            val_metrics = self.validate(model, val_loader, criterion)
            
            # 更新学习率
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['val_acc'].append(val_metrics['accuracy'])
            self.train_history['val_f1'].append(val_metrics['f1'])
            self.train_history['val_auc'].append(val_metrics['auc'])
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/Val', val_metrics['f1'], epoch)
            self.writer.add_scalar('AUC/Val', val_metrics['auc'], epoch)
            
            # 打印结果
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_metrics['loss']:.4f}, 验证准确率: {val_metrics['accuracy']:.4f}")
            print(f"验证F1: {val_metrics['f1']:.4f}, 验证AUC: {val_metrics['auc']:.4f}")
            
            # 保存模型
            is_best = val_metrics['f1'] > best_f1
            if is_best:
                best_f1 = val_metrics['f1']
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_model(model, epoch, val_metrics, is_best)
            
            # 早停检查
            if patience_counter >= self.early_stopping_patience:
                print(f"\n早停触发！已连续 {self.early_stopping_patience} 个epochs无改善")
                break
        
        training_time = time.time() - start_time
        print(f"\n训练完成！总用时: {training_time/60:.2f} 分钟")
        print(f"最佳验证F1分数: {best_f1:.4f}")
        
        # 保存训练历史
        history_path = os.path.join(self.config['paths']['results_path'], 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.train_history, f)
        
        # 关闭TensorBoard
        self.writer.close()
        
        return model, test_loader

def main():
    """主函数"""
    print("PTB心电图异常检测 - 模型训练")
    print("=" * 50)
    
    # 创建训练器
    trainer = ECGTrainer()
    
    # 开始训练
    try:
        model, test_loader = trainer.train()
        print("\n训练成功完成！")
        print("可以运行 evaluate.py 进行模型评估")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()