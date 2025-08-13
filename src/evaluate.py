import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
# torch_geometric import removed
import yaml
import pickle
from typing import Dict, List, Tuple
from tqdm import tqdm

from data_loader import PTBDataLoader


class ECGEvaluator:
    """心电图异常检测评估器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化组件
        self.data_loader = PTBDataLoader(config_path)
        
        
        # 评估配置
        self.eval_config = self.config['evaluation']
        
        # 创建结果目录
        os.makedirs(self.config['paths']['results_path'], exist_ok=True)
    
    def load_model(self, model_path: str = None) -> torch.nn.Module:
        """加载训练好的模型
        
        Args:
            model_path: 模型路径，如果为None则加载最佳模型
            
        Returns:
            model: 加载的模型
        """
        if model_path is None:
            model_path = os.path.join(self.config['paths']['model_save_path'], 'best_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取模型配置
        model_config = checkpoint.get('model_config', {})
        input_dim = self.config['graph']['node_features']
        
        # 创建模型
        model = self.model_factory.create_model(input_dim=input_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"成功加载模型: {model_path}")
        print(f"模型训练epoch: {checkpoint.get('epoch', 'Unknown')}")
        
        return model, checkpoint
    
    def prepare_test_data(self) -> DataLoader:
        """准备测试数据
        
        Returns:
            test_loader: 测试数据加载器
        """
        print("准备测试数据...")
        
        # 加载测试数据
        test_data, test_labels = self.data_loader.load_processed_data('test')
        print(f"测试集: {len(test_data)} 样本")
        
        # 构建图数据

        print(f"测试图: {len(test_graphs)} 个")
        
        # 创建数据加载器
        test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
        
        return test_loader
    
    def predict(self, model: torch.nn.Module, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """模型预测
        
        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器
            
        Returns:
            predictions, probabilities, true_labels: 预测结果、概率和真实标签
        """
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        print("进行模型预测...")
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="预测")
            for batch in pbar:
                batch = batch.to(self.device)
                
                outputs = model(batch)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs), np.array(all_labels)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """计算评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            
        Returns:
            metrics: 评估指标字典
        """
        metrics = {}
        
        # 基本分类指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 每个类别的指标
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['precision_normal'] = precision_per_class[0] if len(precision_per_class) > 0 else 0
        metrics['precision_abnormal'] = precision_per_class[1] if len(precision_per_class) > 1 else 0
        metrics['recall_normal'] = recall_per_class[0] if len(recall_per_class) > 0 else 0
        metrics['recall_abnormal'] = recall_per_class[1] if len(recall_per_class) > 1 else 0
        metrics['f1_normal'] = f1_per_class[0] if len(f1_per_class) > 0 else 0
        metrics['f1_abnormal'] = f1_per_class[1] if len(f1_per_class) > 1 else 0
        
        # AUC指标
        try:
            if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.0
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
        """绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 保存路径
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵保存至: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
        """绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_prob: 预测概率
            save_path: 保存路径
        """
        # 获取异常类的概率
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob
        
        fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
        auc_score = roc_auc_score(y_true, y_prob_pos)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线保存至: {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
        """绘制精确率-召回率曲线
        
        Args:
            y_true: 真实标签
            y_prob: 预测概率
            save_path: 保存路径
        """
        # 获取异常类的概率
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob_pos)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR曲线保存至: {save_path}")
        
        plt.show()
    
    def plot_training_history(self, history_path: str = None, save_path: str = None):
        """绘制训练历史
        
        Args:
            history_path: 训练历史文件路径
            save_path: 保存路径
        """
        if history_path is None:
            history_path = os.path.join(self.config['paths']['results_path'], 'training_history.pkl')
        
        if not os.path.exists(history_path):
            print(f"训练历史文件不存在: {history_path}")
            return
        
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curve
        axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[0, 1].plot(history['train_acc'], label='Training Accuracy', color='blue')
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Accuracy Curve')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 score curve
        axes[1, 0].plot(history['val_f1'], label='Validation F1', color='green')
        axes[1, 0].set_title('F1 Score Curve')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC curve
        axes[1, 1].plot(history['val_auc'], label='Validation AUC', color='purple')
        axes[1, 1].set_title('AUC Curve')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图保存至: {save_path}")
        
        plt.show()
    
    def generate_report(self, metrics: Dict[str, float], y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """生成评估报告
        
        Args:
            metrics: 评估指标
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            report: 评估报告字符串
        """
        report = []
        report.append("=" * 60)
        report.append("PTB心电图异常检测 - 模型评估报告")
        report.append("=" * 60)
        report.append("")
        
        # 整体指标
        report.append("整体性能指标:")
        report.append("-" * 30)
        report.append(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        report.append(f"精确率 (Precision): {metrics['precision']:.4f}")
        report.append(f"召回率 (Recall): {metrics['recall']:.4f}")
        report.append(f"F1分数 (F1-Score): {metrics['f1']:.4f}")
        report.append(f"AUC分数: {metrics['auc']:.4f}")
        report.append("")
        
        # 各类别指标
        report.append("各类别性能指标:")
        report.append("-" * 30)
        report.append("正常类别:")
        report.append(f"  精确率: {metrics['precision_normal']:.4f}")
        report.append(f"  召回率: {metrics['recall_normal']:.4f}")
        report.append(f"  F1分数: {metrics['f1_normal']:.4f}")
        report.append("")
        report.append("异常类别:")
        report.append(f"  精确率: {metrics['precision_abnormal']:.4f}")
        report.append(f"  召回率: {metrics['recall_abnormal']:.4f}")
        report.append(f"  F1分数: {metrics['f1_abnormal']:.4f}")
        report.append("")
        
        # 样本分布
        unique, counts = np.unique(y_true, return_counts=True)
        report.append("测试集样本分布:")
        report.append("-" * 30)
        for label, count in zip(unique, counts):
            class_name = "正常" if label == 0 else "异常"
            percentage = count / len(y_true) * 100
            report.append(f"{class_name}: {count} 样本 ({percentage:.1f}%)")
        report.append("")
        
        # 详细分类报告
        report.append("详细分类报告:")
        report.append("-" * 30)
        class_report = classification_report(y_true, y_pred, 
                                           target_names=['Normal', 'Abnormal'],
                                           digits=4)
        report.append(class_report)
        
        return "\n".join(report)
    
    def evaluate(self, model_path: str = None) -> Dict[str, float]:
        """完整的模型评估流程
        
        Args:
            model_path: 模型路径
            
        Returns:
            metrics: 评估指标
        """
        print("开始模型评估...")
        
        # 加载模型
        model, checkpoint = self.load_model(model_path)
        
        # 准备测试数据
        test_loader = self.prepare_test_data()
        
        # 模型预测
        y_pred, y_prob, y_true = self.predict(model, test_loader)
        
        # 计算指标
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        # 生成报告
        report = self.generate_report(metrics, y_true, y_pred)
        print(report)
        
        # 保存报告
        report_path = os.path.join(self.config['paths']['results_path'], 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n评估报告保存至: {report_path}")
        
        # 保存指标
        metrics_path = os.path.join(self.config['paths']['results_path'], 'evaluation_metrics.pkl')
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        
        # 绘制图表
        results_path = self.config['paths']['results_path']
        
        if self.eval_config.get('save_confusion_matrix', True):
            cm_path = os.path.join(results_path, 'confusion_matrix.png')
            self.plot_confusion_matrix(y_true, y_pred, cm_path)
        
        if self.eval_config.get('save_roc_curve', True):
            roc_path = os.path.join(results_path, 'roc_curve.png')
            self.plot_roc_curve(y_true, y_prob, roc_path)
        
        # 绘制PR曲线
        pr_path = os.path.join(results_path, 'precision_recall_curve.png')
        self.plot_precision_recall_curve(y_true, y_prob, pr_path)
        
        # 绘制训练历史
        history_path = os.path.join(results_path, 'training_history.png')
        self.plot_training_history(save_path=history_path)
        
        print("\n评估完成！")
        return metrics

def main():
    """主函数"""
    print("PTB心电图异常检测 - 模型评估")
    print("=" * 50)
    
    # 创建评估器
    evaluator = ECGEvaluator()
    
    try:
        # 开始评估
        metrics = evaluator.evaluate()
        
        print(f"\n评估完成！最终F1分数: {metrics['f1']:.4f}")
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请先运行 train.py 训练模型")
    except Exception as e:
        print(f"\n评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()