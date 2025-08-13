import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import time
import os
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入模型
from comparison_models import create_comparison_model
from data_loader import PTBDataLoader

class ModelComparison:
    """模型比较类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # 创建结果保存目录
        self.results_dir = "results/comparison"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")
        
    def prepare_data(self):
        """准备数据"""
        print("Preparing data...")
        
        # 加载数据处理器
        data_processor = ECGDataProcessor(self.config)
        
        # 检查是否存在处理后的数据
        processed_data_path = self.config['data']['processed_data_path']
        if not os.path.exists(processed_data_path):
            print("Processed data not found. Please run data preprocessing first.")
            return None, None, None
            
        # 加载处理后的数据
        data = np.load(processed_data_path, allow_pickle=True).item()
        
        # 提取数据
        X_train = data['X_train']
        X_val = data['X_val'] 
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        
        print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)
        
    def prepare_graph_data(self):
        """准备图数据"""
        print("Preparing graph data...")
        
        # 检查图数据是否存在
        graph_data_paths = {
            'train': self.config['data']['train_graph_path'],
            'val': self.config['data']['val_graph_path'],
            'test': self.config['data']['test_graph_path']
        }
        
        graph_data = {}
        for split, path in graph_data_paths.items():
            if os.path.exists(path):
                graph_data[split] = torch.load(path)
            else:
                print(f"Graph data not found: {path}")
                return None
                
        return graph_data
        
    def create_dataloaders(self, X_data, y_data, batch_size: int = 32):
        """创建数据加载器"""
        X_train, X_val, X_test = X_data
        y_train, y_val, y_test = y_data
        
        # 转换为张量
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        X_test = torch.FloatTensor(X_test)
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
        
    def train_model(self, model, train_loader, val_loader, model_name: str, 
                   num_epochs: int = 50, learning_rate: float = 0.001):
        """训练模型"""
        print(f"\nTraining {model_name} model...")
        
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        factor=0.5, patience=5)
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # 调整输入格式
                if model_name in ['LSTM', 'Transformer']:
                    batch_X = batch_X.transpose(1, 2)  # (batch, seq_len, features)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            # 验证阶段
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    if model_name in ['LSTM', 'Transformer']:
                        batch_X = batch_X.transpose(1, 2)
                        
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)
            
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 
                          os.path.join(self.results_dir, f"best_{model_name.lower()}_model.pth"))
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        training_time = time.time() - start_time
        
        # 加载最佳模型
        model.load_state_dict(torch.load(os.path.join(self.results_dir, 
                                                      f"best_{model_name.lower()}_model.pth")))
        
        return model, {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc,
            'training_time': training_time
        }
        
    def evaluate_model(self, model, test_loader, model_name: str):
        """评估模型"""
        print(f"\nEvaluating {model_name} model...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                if model_name in ['LSTM', 'Transformer']:
                    batch_X = batch_X.transpose(1, 2)
                    
                outputs = model(batch_X)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
        inference_time = time.time() - start_time
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # 计算AUC（二分类）
        if len(np.unique(all_labels)) == 2:
            auc = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
            
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'inference_time': inference_time,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        

        
    def run_comparison(self):
        """运行模型比较"""
        print("Starting model comparison...")
        
        # 准备数据
        X_data, y_data = self.prepare_data()
        if X_data is None:
            return
            
        graph_data = self.prepare_graph_data()
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_dataloaders(X_data, y_data, batch_size=32)
        
        # 定义要比较的模型
        models_to_compare = {
            'CNN': {'input_length': X_data[0].shape[2], 'num_classes': 2},
            'LSTM': {'input_size': X_data[0].shape[1], 'num_classes': 2},
            'Transformer': {'input_size': X_data[0].shape[1], 'num_classes': 2},
            'ResNet': {'num_classes': 2},
            'CNN-LSTM': {'num_classes': 2}
        }
        
        # 训练和评估每个模型
        for model_name, model_params in models_to_compare.items():
            try:
                print(f"\n{'='*50}")
                print(f"Processing {model_name} Model")
                print(f"{'='*50}")
                
                # 创建模型
                model = create_comparison_model(model_name, **model_params)
                print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
                
                # 训练模型
                trained_model, training_info = self.train_model(
                    model, train_loader, val_loader, model_name
                )
                
                # 评估模型
                eval_results = self.evaluate_model(trained_model, test_loader, model_name)
                
                # 保存结果
                self.results[model_name] = {
                    **eval_results,
                    **training_info,
                    'num_parameters': sum(p.numel() for p in model.parameters())
                }
                
                print(f"{model_name} Results:")
                print(f"  Accuracy: {eval_results['accuracy']:.4f}")
                print(f"  F1 Score: {eval_results['f1_score']:.4f}")
                print(f"  AUC Score: {eval_results['auc_score']:.4f}")
                print(f"  Training Time: {training_info['training_time']:.2f}s")
                print(f"  Inference Time: {eval_results['inference_time']:.2f}s")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
                

                
        # 生成比较报告
        self.generate_comparison_report()
        
    def generate_comparison_report(self):
        """生成比较报告"""
        print("\nGenerating comparison report...")
        
        if not self.results:
            print("No results to compare.")
            return
            
        # 创建结果DataFrame
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        comparison_data = []
        
        for model_name, results in self.results.items():
            row = {'Model': model_name}
            for metric in metrics:
                if metric in results:
                    row[metric.replace('_', ' ').title()] = results[metric]
                    
            if 'training_time' in results:
                row['Training Time (s)'] = results['training_time']
            if 'inference_time' in results:
                row['Inference Time (s)'] = results['inference_time']
            if 'num_parameters' in results:
                row['Parameters'] = results['num_parameters']
                
            comparison_data.append(row)
            
        df = pd.DataFrame(comparison_data)
        
        # 保存结果表格
        df.to_csv(os.path.join(self.results_dir, 'model_comparison_results.csv'), index=False)
        
        # 打印结果表格
        print("\nModel Comparison Results:")
        print(df.to_string(index=False, float_format='%.4f'))
        
        # 生成可视化
        self.plot_comparison_results(df)
        
        # 找出最佳模型
        best_model_f1 = df.loc[df['F1 Score'].idxmax(), 'Model']
        best_model_auc = df.loc[df['Auc Score'].idxmax(), 'Model']
        
        print(f"\nBest Model (F1 Score): {best_model_f1} ({df['F1 Score'].max():.4f})")
        print(f"Best Model (AUC Score): {best_model_auc} ({df['Auc Score'].max():.4f})")
        
    def plot_comparison_results(self, df):
        """绘制比较结果"""
        # 设置图形样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Comparison Results', fontsize=16, fontweight='bold')
        
        # 性能指标比较
        metrics = ['Accuracy', 'F1 Score', 'Auc Score']
        ax1 = axes[0, 0]
        x = np.arange(len(df))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                ax1.bar(x + i*width, df[metric], width, label=metric, alpha=0.8)
                
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics Comparison')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(df['Model'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 训练时间比较
        ax2 = axes[0, 1]
        if 'Training Time (s)' in df.columns:
            bars = ax2.bar(df['Model'], df['Training Time (s)'], alpha=0.7, color='skyblue')
            ax2.set_xlabel('Models')
            ax2.set_ylabel('Training Time (seconds)')
            ax2.set_title('Training Time Comparison')
            ax2.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}s', ha='center', va='bottom')
                        
        # 推理时间比较
        ax3 = axes[1, 0]
        if 'Inference Time (s)' in df.columns:
            bars = ax3.bar(df['Model'], df['Inference Time (s)'], alpha=0.7, color='lightcoral')
            ax3.set_xlabel('Models')
            ax3.set_ylabel('Inference Time (seconds)')
            ax3.set_title('Inference Time Comparison')
            ax3.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}s', ha='center', va='bottom')
                        
        # 参数数量比较
        ax4 = axes[1, 1]
        if 'Parameters' in df.columns:
            bars = ax4.bar(df['Model'], df['Parameters']/1000, alpha=0.7, color='lightgreen')
            ax4.set_xlabel('Models')
            ax4.set_ylabel('Parameters (K)')
            ax4.set_title('Model Size Comparison')
            ax4.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}K', ha='center', va='bottom')
                        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 创建雷达图
        self.plot_radar_chart(df)
        
    def plot_radar_chart(self, df):
        """绘制雷达图"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Auc Score']
        available_metrics = [m for m in metrics if m in df.columns]
        
        if len(available_metrics) < 3:
            return
            
        # 设置雷达图
        angles = np.linspace(0, 2*np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
        
        for i, (_, row) in enumerate(df.iterrows()):
            values = [row[metric] for metric in available_metrics]
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_radar_chart.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    print("ECG Classification Model Comparison")
    print("=" * 50)
    
    # 创建比较器
    comparator = ModelComparison()
    
    # 运行比较
    comparator.run_comparison()
    
    print("\nComparison completed! Results saved in results/comparison/")

if __name__ == "__main__":
    main()