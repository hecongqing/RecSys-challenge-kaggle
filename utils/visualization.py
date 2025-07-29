"""
RecSys Challenge 2015 可视化工具模块

这个模块提供额外的可视化功能：
1. 图结构可视化
2. 模型解释性分析
3. 数据分布可视化
4. 性能对比图表
5. 交互式可视化

作者: GNN Tutorial
日期: 2024
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from typing import List, Dict, Optional, Tuple
import os


class GraphVisualizer:
    """
    图可视化工具类
    
    用于可视化会话图结构和分析
    """
    
    @staticmethod
    def visualize_session_graph(data: Data, 
                               session_id: str = "Session",
                               item_names: Optional[Dict] = None,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 8)):
        """
        可视化单个会话图
        
        Args:
            data (Data): PyTorch Geometric图数据
            session_id (str): 会话ID
            item_names (Dict, optional): 物品ID到名称的映射
            save_path (str, optional): 保存路径
            figsize (tuple): 图形大小
        """
        # 转换为NetworkX图
        G = to_networkx(data, to_undirected=False)
        
        # 设置图形
        plt.figure(figsize=figsize)
        
        # 布局
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 绘制节点
        node_colors = []
        node_labels = {}
        
        for i, node in enumerate(G.nodes()):
            item_id = data.x[node].item()
            
            # 节点颜色（根据物品ID）
            node_colors.append(item_id % 10)  # 简单着色
            
            # 节点标签
            if item_names and item_id in item_names:
                node_labels[node] = item_names[item_id]
            else:
                node_labels[node] = f"Item_{item_id}"
        
        # 绘制图
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors, 
                              node_size=800,
                              cmap=plt.cm.Set3,
                              alpha=0.8)
        
        nx.draw_networkx_edges(G, pos, 
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              alpha=0.6,
                              arrowstyle='->')
        
        nx.draw_networkx_labels(G, pos, 
                               node_labels,
                               font_size=8,
                               font_weight='bold')
        
        plt.title(f"{session_id} - 会话图结构\n"
                 f"节点数: {data.num_nodes}, 边数: {data.num_edges}, "
                 f"标签: {'购买' if data.y.item() == 1 else '未购买'}")
        
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"会话图已保存: {save_path}")
        
        plt.show()
    
    @staticmethod
    def visualize_multiple_sessions(dataset, 
                                   num_sessions: int = 6,
                                   save_path: Optional[str] = None):
        """
        可视化多个会话图
        
        Args:
            dataset: 数据集
            num_sessions (int): 要可视化的会话数量
            save_path (str, optional): 保存路径
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 随机选择会话
        indices = np.random.choice(len(dataset), num_sessions, replace=False)
        
        for i, idx in enumerate(indices):
            data = dataset[idx]
            G = to_networkx(data, to_undirected=False)
            
            # 布局
            pos = nx.spring_layout(G, k=1, iterations=30)
            
            # 节点颜色
            node_colors = [data.x[node].item() % 10 for node in G.nodes()]
            
            # 绘制在子图上
            ax = axes[i]
            nx.draw(G, pos, ax=ax,
                   node_color=node_colors,
                   node_size=300,
                   cmap=plt.cm.Set3,
                   with_labels=True,
                   labels={n: f"{data.x[n].item()}" for n in G.nodes()},
                   font_size=6,
                   arrows=True,
                   arrowsize=10,
                   edge_color='gray',
                   alpha=0.7)
            
            ax.set_title(f"会话 {idx}\n"
                        f"节点: {data.num_nodes}, 边: {data.num_edges}\n"
                        f"标签: {'购买' if data.y.item() == 1 else '未购买'}")
            ax.axis('off')
        
        plt.suptitle("多个会话图结构示例", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"多会话图已保存: {save_path}")
        
        plt.show()


class DataDistributionVisualizer:
    """
    数据分布可视化工具类
    """
    
    @staticmethod
    def plot_session_length_distribution(dataset, 
                                        save_path: Optional[str] = None):
        """
        绘制会话长度分布
        
        Args:
            dataset: 数据集
            save_path (str, optional): 保存路径
        """
        session_lengths = [data.num_nodes for data in dataset]
        
        plt.figure(figsize=(12, 5))
        
        # 直方图
        plt.subplot(1, 2, 1)
        plt.hist(session_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('会话长度（物品数量）')
        plt.ylabel('频次')
        plt.title('会话长度分布')
        plt.grid(True, alpha=0.3)
        
        # 箱线图
        plt.subplot(1, 2, 2)
        plt.boxplot(session_lengths, vert=True)
        plt.ylabel('会话长度（物品数量）')
        plt.title('会话长度箱线图')
        plt.grid(True, alpha=0.3)
        
        # 统计信息
        stats_text = f"平均长度: {np.mean(session_lengths):.2f}\n"
        stats_text += f"中位数: {np.median(session_lengths):.2f}\n"
        stats_text += f"标准差: {np.std(session_lengths):.2f}\n"
        stats_text += f"最小值: {np.min(session_lengths)}\n"
        stats_text += f"最大值: {np.max(session_lengths)}"
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"会话长度分布图已保存: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_label_distribution(dataset, 
                               save_path: Optional[str] = None):
        """
        绘制标签分布
        
        Args:
            dataset: 数据集
            save_path (str, optional): 保存路径
        """
        labels = [data.y.item() for data in dataset]
        
        plt.figure(figsize=(10, 6))
        
        # 计算比例
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        percentages = counts / total * 100
        
        # 饼图
        plt.subplot(1, 2, 1)
        colors = ['lightcoral', 'lightblue']
        labels_text = ['未购买', '购买']
        
        plt.pie(counts, labels=labels_text, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=[0.05, 0])
        plt.title('购买标签分布')
        
        # 柱状图
        plt.subplot(1, 2, 2)
        bars = plt.bar(labels_text, counts, color=colors, alpha=0.7, edgecolor='black')
        plt.xlabel('标签')
        plt.ylabel('数量')
        plt.title('购买标签统计')
        
        # 添加数值标签
        for bar, count, pct in zip(bars, counts, percentages):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"标签分布图已保存: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_item_popularity(dataset, 
                           top_k: int = 20,
                           save_path: Optional[str] = None):
        """
        绘制物品流行度分布
        
        Args:
            dataset: 数据集
            top_k (int): 显示top-k物品
            save_path (str, optional): 保存路径
        """
        # 收集所有物品
        all_items = []
        for data in dataset:
            all_items.extend(data.x.flatten().tolist())
        
        # 计算物品频次
        unique_items, counts = np.unique(all_items, return_counts=True)
        
        # 排序并取top-k
        sorted_indices = np.argsort(counts)[::-1]
        top_items = unique_items[sorted_indices[:top_k]]
        top_counts = counts[sorted_indices[:top_k]]
        
        plt.figure(figsize=(15, 8))
        
        # 流行物品柱状图
        plt.subplot(2, 1, 1)
        bars = plt.bar(range(len(top_items)), top_counts, color='steelblue', alpha=0.7)
        plt.xlabel('物品排名')
        plt.ylabel('出现次数')
        plt.title(f'Top-{top_k} 热门物品')
        plt.xticks(range(len(top_items)), [f'Item_{item}' for item in top_items], 
                  rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, (bar, count) in enumerate(zip(bars, top_counts)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(top_counts)*0.01,
                    str(count), ha='center', va='bottom', fontsize=8)
        
        # 物品频次分布
        plt.subplot(2, 1, 2)
        plt.hist(counts, bins=50, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('物品出现次数')
        plt.ylabel('物品数量')
        plt.title('物品流行度分布')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"物品流行度图已保存: {save_path}")
        
        plt.show()


class ModelAnalysisVisualizer:
    """
    模型分析可视化工具类
    """
    
    @staticmethod
    def plot_embedding_analysis(model, 
                               num_items: int,
                               method: str = 'tsne',
                               save_path: Optional[str] = None):
        """
        分析和可视化物品嵌入
        
        Args:
            model: 训练好的模型
            num_items (int): 物品数量
            method (str): 降维方法 ('tsne', 'pca')
            save_path (str, optional): 保存路径
        """
        # 获取嵌入权重
        embedding_weights = model.item_embedding.weight.detach().cpu().numpy()
        
        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, num_items-1))
            reduced_embeddings = reducer.fit_transform(embedding_weights)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            reduced_embeddings = reducer.fit_transform(embedding_weights)
        else:
            raise ValueError(f"不支持的降维方法: {method}")
        
        plt.figure(figsize=(12, 8))
        
        # 散点图
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                            c=range(num_items), cmap='viridis', alpha=0.6, s=50)
        
        # 添加一些物品标签
        sample_indices = np.random.choice(num_items, min(20, num_items), replace=False)
        for idx in sample_indices:
            plt.annotate(f'Item_{idx}', 
                        (reduced_embeddings[idx, 0], reduced_embeddings[idx, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        plt.colorbar(scatter, label='物品ID')
        plt.xlabel(f'{method.upper()} 维度 1')
        plt.ylabel(f'{method.upper()} 维度 2')
        plt.title(f'物品嵌入可视化 ({method.upper()})')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"嵌入可视化图已保存: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_attention_analysis(model, 
                               data_sample: Data,
                               save_path: Optional[str] = None):
        """
        分析注意力权重（如果模型支持）
        
        Args:
            model: 带注意力机制的模型
            data_sample (Data): 数据样本
            save_path (str, optional): 保存路径
        """
        if not hasattr(model, 'attention'):
            print("模型不包含注意力机制")
            return
        
        model.eval()
        with torch.no_grad():
            # 这里需要根据具体的注意力实现来获取权重
            # 这是一个示例实现
            print("注意力分析功能需要根据具体模型实现")
            print("请参考AttentionSessionGNN模型的实现")


class InteractiveVisualizer:
    """
    交互式可视化工具类（使用Plotly）
    """
    
    @staticmethod
    def create_interactive_training_plot(history: Dict, 
                                       save_path: Optional[str] = None):
        """
        创建交互式训练历史图
        
        Args:
            history (Dict): 训练历史
            save_path (str, optional): 保存路径
        """
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('损失曲线', 'AUC曲线', '学习率变化', '性能对比'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # 损失曲线
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_loss'], 
                      name='训练损失', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_loss'], 
                      name='验证损失', line=dict(color='red')),
            row=1, col=1
        )
        
        # AUC曲线
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_auc'], 
                      name='训练AUC', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_auc'], 
                      name='验证AUC', line=dict(color='red')),
            row=1, col=2
        )
        
        # 学习率变化
        fig.add_trace(
            go.Scatter(x=epochs, y=history['lr'], 
                      name='学习率', line=dict(color='green')),
            row=2, col=1
        )
        
        # 性能对比
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_auc'], 
                      name='训练AUC', fill='tonexty', 
                      line=dict(color='blue'), opacity=0.6),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_auc'], 
                      name='验证AUC', fill='tonexty',
                      line=dict(color='red'), opacity=0.6),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title_text="交互式训练历史",
            showlegend=True,
            height=600
        )
        
        # 设置y轴为对数刻度（学习率）
        fig.update_yaxes(type="log", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            print(f"交互式图表已保存: {save_path}")
        
        fig.show()
    
    @staticmethod
    def create_model_comparison_plot(results_dict: Dict[str, Dict],
                                   save_path: Optional[str] = None):
        """
        创建模型对比图
        
        Args:
            results_dict (Dict): 不同模型的结果字典
            save_path (str, optional): 保存路径
        """
        metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1']
        model_names = list(results_dict.keys())
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [results_dict[model].get(metric, 0) for model in model_names]
            
            fig.add_trace(go.Scatter(
                x=model_names,
                y=values,
                mode='lines+markers',
                name=metric.upper(),
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="模型性能对比",
            xaxis_title="模型类型",
            yaxis_title="性能指标",
            yaxis=dict(range=[0, 1]),
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"模型对比图已保存: {save_path}")
        
        fig.show()


def create_comprehensive_report(dataset, 
                              model, 
                              results, 
                              save_dir: str):
    """
    创建综合可视化报告
    
    Args:
        dataset: 数据集
        model: 模型
        results: 训练结果
        save_dir (str): 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("正在生成综合可视化报告...")
    
    # 1. 数据分布分析
    DataDistributionVisualizer.plot_session_length_distribution(
        dataset, os.path.join(save_dir, 'session_length_dist.png')
    )
    
    DataDistributionVisualizer.plot_label_distribution(
        dataset, os.path.join(save_dir, 'label_distribution.png')
    )
    
    DataDistributionVisualizer.plot_item_popularity(
        dataset, save_path=os.path.join(save_dir, 'item_popularity.png')
    )
    
    # 2. 图结构可视化
    GraphVisualizer.visualize_multiple_sessions(
        dataset, save_path=os.path.join(save_dir, 'session_graphs.png')
    )
    
    # 3. 模型分析
    num_items = model.num_items
    ModelAnalysisVisualizer.plot_embedding_analysis(
        model, num_items, method='tsne',
        save_path=os.path.join(save_dir, 'embedding_tsne.png')
    )
    
    ModelAnalysisVisualizer.plot_embedding_analysis(
        model, num_items, method='pca',
        save_path=os.path.join(save_dir, 'embedding_pca.png')
    )
    
    # 4. 交互式图表
    if 'history' in results:
        InteractiveVisualizer.create_interactive_training_plot(
            results['history'], 
            save_path=os.path.join(save_dir, 'interactive_training.html')
        )
    
    print(f"综合可视化报告已生成: {save_dir}")


def main():
    """
    主函数：演示可视化功能
    """
    print("=== RecSys Challenge 2015 可视化工具演示 ===")
    print("这是可视化工具模块")
    print("请通过main.py使用完整的可视化功能")


if __name__ == "__main__":
    main()