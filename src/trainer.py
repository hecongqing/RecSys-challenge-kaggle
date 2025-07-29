"""
RecSys Challenge 2015 训练和评估模块

这个模块负责：
1. 模型训练循环
2. 验证和测试评估
3. 指标计算和分析
4. 模型保存和加载
5. 早停和学习率调度

评估指标：
- AUC (Area Under Curve)：主要评估指标
- Accuracy：准确率
- Precision & Recall：精确率和召回率
- F1-Score：F1分数

作者: GNN Tutorial
日期: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class SessionTrainer:
    """
    会话推荐模型训练器
    
    负责模型的训练、验证、测试和结果分析
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 device: torch.device,
                 config: Optional[Dict] = None):
        """
        初始化训练器
        
        Args:
            model (nn.Module): 要训练的模型
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader): 验证数据加载器
            test_loader (DataLoader): 测试数据加载器
            device (torch.device): 计算设备
            config (Dict): 训练配置
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # 默认配置
        default_config = {
            'learning_rate': 0.005,
            'weight_decay': 1e-4,
            'num_epochs': 100,
            'patience': 10,
            'min_delta': 1e-4,
            'save_dir': 'models',
            'log_interval': 100,
            'save_best_only': True
        }
        
        if config is None:
            config = {}
        default_config.update(config)
        self.config = default_config
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.criterion = nn.BCELoss()  # 二分类损失
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_auc': [],
            'val_loss': [],
            'val_auc': [],
            'lr': []
        }
        
        # 早停
        self.best_val_auc = 0.0
        self.patience_counter = 0
        
        # 确保保存目录存在
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
    def train_epoch(self) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Returns:
            Tuple[float, float]: 训练损失和AUC
        """
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # 训练循环
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="训练中")):
            batch = batch.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch.y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * batch.num_graphs
            all_predictions.extend(outputs.detach().cpu().numpy())
            all_labels.extend(batch.y.detach().cpu().numpy())
            
            # 日志输出
            if batch_idx % self.config['log_interval'] == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        # 计算平均损失和AUC
        avg_loss = total_loss / len(self.train_loader.dataset)
        auc = roc_auc_score(all_labels, all_predictions)
        
        return avg_loss, auc
    
    def validate(self, loader: DataLoader, desc: str = "验证中") -> Tuple[float, float, Dict]:
        """
        验证模型
        
        Args:
            loader (DataLoader): 数据加载器
            desc (str): 进度条描述
            
        Returns:
            Tuple[float, float, Dict]: 损失、AUC和详细指标
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                batch = batch.to(self.device)
                
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch.y)
                
                total_loss += loss.item() * batch.num_graphs
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / len(loader.dataset)
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # 计算AUC
        auc = roc_auc_score(all_labels, all_predictions)
        
        # 计算其他指标（使用0.5作为阈值）
        pred_binary = (all_predictions >= 0.5).astype(int)
        accuracy = accuracy_score(all_labels, pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, pred_binary, average='binary'
        )
        
        metrics = {
            'loss': avg_loss,
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        return avg_loss, auc, metrics
    
    def train(self) -> Dict:
        """
        完整的训练过程
        
        Returns:
            Dict: 训练历史和最终结果
        """
        print(f"开始训练模型，设备: {self.device}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print(f"测试集大小: {len(self.test_loader.dataset)}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print("-" * 30)
            
            # 训练
            train_loss, train_auc = self.train_epoch()
            
            # 验证
            val_loss, val_auc, val_metrics = self.validate(self.val_loader, "验证中")
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # 学习率调度
            self.scheduler.step(val_auc)
            
            # 输出结果
            epoch_time = time.time() - epoch_start
            print(f"训练损失: {train_loss:.4f}, 训练AUC: {train_auc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证AUC: {val_auc:.4f}")
            print(f"验证准确率: {val_metrics['accuracy']:.4f}")
            print(f"Epoch时间: {epoch_time:.2f}秒")
            
            # 早停检查
            if val_auc > self.best_val_auc + self.config['min_delta']:
                self.best_val_auc = val_auc
                self.patience_counter = 0
                
                # 保存最佳模型
                if self.config['save_best_only']:
                    self.save_model('best_model.pth', epoch, val_auc)
                    print(f"保存最佳模型，验证AUC: {val_auc:.4f}")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config['patience']:
                print(f"早停：{self.config['patience']} 个epoch没有改善")
                break
            
            # 保存检查点
            if not self.config['save_best_only']:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_auc)
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总时间: {total_time:.2f}秒")
        
        # 在最佳模型上测试
        if self.config['save_best_only']:
            self.load_model('best_model.pth')
        
        test_loss, test_auc, test_metrics = self.validate(self.test_loader, "测试中")
        
        print(f"\n=== 最终测试结果 ===")
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试AUC: {test_auc:.4f}")
        print(f"测试准确率: {test_metrics['accuracy']:.4f}")
        print(f"测试精确率: {test_metrics['precision']:.4f}")
        print(f"测试召回率: {test_metrics['recall']:.4f}")
        print(f"测试F1分数: {test_metrics['f1']:.4f}")
        
        # 返回结果
        results = {
            'history': self.history,
            'best_val_auc': self.best_val_auc,
            'test_metrics': test_metrics,
            'total_time': total_time
        }
        
        # 保存训练历史
        self.save_history()
        
        return results
    
    def save_model(self, filename: str, epoch: int, val_auc: float):
        """
        保存模型
        
        Args:
            filename (str): 文件名
            epoch (int): 当前epoch
            val_auc (float): 验证AUC
        """
        filepath = os.path.join(self.config['save_dir'], filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_auc': val_auc,
            'config': self.config,
            'history': self.history
        }, filepath)
    
    def load_model(self, filename: str):
        """
        加载模型
        
        Args:
            filename (str): 文件名
        """
        filepath = os.path.join(self.config['save_dir'], filename)
        
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.history = checkpoint['history']
            print(f"模型加载成功: {filepath}")
        else:
            print(f"模型文件不存在: {filepath}")
    
    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.config['save_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"训练历史已保存: {history_path}")


class MetricsAnalyzer:
    """
    指标分析器
    
    用于分析和可视化模型性能
    """
    
    @staticmethod
    def plot_training_history(history: Dict, save_path: Optional[str] = None):
        """
        绘制训练历史
        
        Args:
            history (Dict): 训练历史
            save_path (str, optional): 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(history['train_loss'], label='训练损失', color='blue')
        axes[0, 0].plot(history['val_loss'], label='验证损失', color='red')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # AUC曲线
        axes[0, 1].plot(history['train_auc'], label='训练AUC', color='blue')
        axes[0, 1].plot(history['val_auc'], label='验证AUC', color='red')
        axes[0, 1].set_title('AUC曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线
        axes[1, 0].plot(history['lr'], label='学习率', color='green')
        axes[1, 0].set_title('学习率变化')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
        
        # 训练进度
        epochs = range(1, len(history['train_loss']) + 1)
        axes[1, 1].plot(epochs, history['train_auc'], label='训练AUC', color='blue')
        axes[1, 1].plot(epochs, history['val_auc'], label='验证AUC', color='red')
        axes[1, 1].fill_between(epochs, history['train_auc'], alpha=0.3, color='blue')
        axes[1, 1].fill_between(epochs, history['val_auc'], alpha=0.3, color='red')
        axes[1, 1].set_title('AUC对比')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(labels: np.ndarray, 
                            predictions: np.ndarray, 
                            threshold: float = 0.5,
                            save_path: Optional[str] = None):
        """
        绘制混淆矩阵
        
        Args:
            labels (np.ndarray): 真实标签
            predictions (np.ndarray): 预测概率
            threshold (float): 分类阈值
            save_path (str, optional): 保存路径
        """
        pred_binary = (predictions >= threshold).astype(int)
        cm = confusion_matrix(labels, pred_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['无购买', '有购买'],
                   yticklabels=['无购买', '有购买'])
        plt.title(f'混淆矩阵 (阈值: {threshold})')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存: {save_path}")
        
        plt.show()
    
    @staticmethod
    def analyze_predictions(labels: np.ndarray, 
                          predictions: np.ndarray) -> pd.DataFrame:
        """
        分析预测结果
        
        Args:
            labels (np.ndarray): 真实标签
            predictions (np.ndarray): 预测概率
            
        Returns:
            pd.DataFrame: 分析结果
        """
        # 不同阈值下的性能
        thresholds = np.arange(0.1, 1.0, 0.1)
        results = []
        
        for threshold in thresholds:
            pred_binary = (predictions >= threshold).astype(int)
            
            accuracy = accuracy_score(labels, pred_binary)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, pred_binary, average='binary'
            )
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def plot_prediction_distribution(labels: np.ndarray,
                                   predictions: np.ndarray,
                                   save_path: Optional[str] = None):
        """
        绘制预测概率分布
        
        Args:
            labels (np.ndarray): 真实标签
            predictions (np.ndarray): 预测概率
            save_path (str, optional): 保存路径
        """
        plt.figure(figsize=(12, 5))
        
        # 分离正负样本的预测
        pos_pred = predictions[labels == 1]
        neg_pred = predictions[labels == 0]
        
        # 绘制分布
        plt.subplot(1, 2, 1)
        plt.hist(neg_pred, bins=50, alpha=0.7, label='无购买', color='blue', density=True)
        plt.hist(pos_pred, bins=50, alpha=0.7, label='有购买', color='red', density=True)
        plt.xlabel('预测概率')
        plt.ylabel('密度')
        plt.title('预测概率分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制ROC曲线相关的累积分布
        plt.subplot(1, 2, 2)
        plt.hist(neg_pred, bins=50, alpha=0.7, label='无购买', color='blue', cumulative=True, density=True)
        plt.hist(pos_pred, bins=50, alpha=0.7, label='有购买', color='red', cumulative=True, density=True)
        plt.xlabel('预测概率')
        plt.ylabel('累积密度')
        plt.title('累积预测概率分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测分布图已保存: {save_path}")
        
        plt.show()


def main():
    """
    主函数：演示训练流程
    """
    print("=== RecSys Challenge 2015 训练演示 ===\n")
    
    # 这个演示需要先运行数据预处理和模型创建
    print("注意：这是训练模块的演示")
    print("实际使用时需要先运行数据预处理和模型创建")
    print("请参考 main.py 获取完整的训练流程")


if __name__ == "__main__":
    main()