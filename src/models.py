"""
RecSys Challenge 2015 图神经网络模型模块

这个模块包含：
1. 基础的GNN模型架构
2. 多种图神经网络层的组合
3. 图级别的分类器
4. 模型工具函数

模型架构说明：
- 使用SAGEConv进行图卷积
- 使用TopKPooling进行图池化
- 使用全局池化聚合节点特征
- 最终进行二分类预测

作者: GNN Tutorial
日期: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, TopKPooling, Set2Set
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Batch
from typing import Optional, Dict, Any
import math


class SessionGNN(nn.Module):
    """
    基于图神经网络的会话推荐模型
    
    架构组成：
    1. 物品嵌入层
    2. 多层图卷积网络
    3. 图池化层
    4. 全局池化和特征聚合
    5. 分类预测头
    """
    
    def __init__(self, 
                 num_items: int,
                 embed_dim: int = 128,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 pooling_ratio: float = 0.8,
                 dropout: float = 0.5,
                 conv_type: str = 'SAGEConv'):
        """
        初始化会话GNN模型
        
        Args:
            num_items (int): 物品总数
            embed_dim (int): 嵌入维度
            hidden_dim (int): 隐藏层维度
            num_layers (int): GNN层数
            pooling_ratio (float): TopK池化比例
            dropout (float): Dropout概率
            conv_type (str): 图卷积类型 ('SAGEConv', 'GCNConv', 'GATConv')
        """
        super(SessionGNN, self).__init__()
        
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 1. 物品嵌入层
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items, 
            embedding_dim=embed_dim,
            padding_idx=0  # 0作为padding
        )
        
        # 初始化嵌入权重
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # 2. 图卷积层
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # 输入维度：第一层是embed_dim，后续层是hidden_dim
            in_dim = embed_dim if i == 0 else hidden_dim
            
            # 图卷积层
            if conv_type == 'SAGEConv':
                conv = SAGEConv(in_dim, hidden_dim, normalize=True)
            elif conv_type == 'GCNConv':
                conv = GCNConv(in_dim, hidden_dim, normalize=True)
            elif conv_type == 'GATConv':
                conv = GATConv(in_dim, hidden_dim, heads=1, concat=False)
            else:
                raise ValueError(f"不支持的卷积类型: {conv_type}")
            
            self.convs.append(conv)
            
            # TopK池化层
            pool = TopKPooling(hidden_dim, ratio=pooling_ratio)
            self.pools.append(pool)
            
            # 批归一化
            bn = nn.BatchNorm1d(hidden_dim)
            self.batch_norms.append(bn)
        
        # 3. 全局特征聚合
        # 每层的全局特征维度：hidden_dim * 2 (mean + max pooling)
        self.global_feature_dim = hidden_dim * 2 * num_layers
        
        # 4. 分类预测头
        self.classifier = nn.Sequential(
            nn.Linear(self.global_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, data):
        """
        前向传播
        
        Args:
            data: PyTorch Geometric数据批次
            
        Returns:
            torch.Tensor: 预测结果 [batch_size, 1]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. 物品嵌入
        # x: [num_nodes_in_batch, 1] -> [num_nodes_in_batch, embed_dim]
        x = self.item_embedding(x.squeeze(-1))
        
        # 2. 多层图卷积和池化
        global_features = []
        
        for i in range(self.num_layers):
            # 图卷积
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # TopK池化
            x, edge_index, _, batch, _ = self.pools[i](
                x, edge_index, batch=batch
            )
            
            # 全局池化：获取图级别特征
            global_mean = global_mean_pool(x, batch)
            global_max = global_max_pool(x, batch)
            global_feat = torch.cat([global_mean, global_max], dim=1)
            global_features.append(global_feat)
        
        # 3. 特征聚合：连接所有层的全局特征
        # 这样可以利用不同层次的信息
        x = torch.cat(global_features, dim=1)  # [batch_size, global_feature_dim]
        
        # 4. 分类预测
        x = self.classifier(x)
        x = torch.sigmoid(x)  # 输出概率
        
        return x.squeeze(-1)  # [batch_size]


class AttentionSessionGNN(nn.Module):
    """
    带注意力机制的会话GNN模型
    
    增加了注意力机制来更好地聚合不同层的特征
    """
    
    def __init__(self, 
                 num_items: int,
                 embed_dim: int = 128,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 pooling_ratio: float = 0.8,
                 dropout: float = 0.5):
        super(AttentionSessionGNN, self).__init__()
        
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 物品嵌入
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # 图卷积层
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            conv = SAGEConv(in_dim, hidden_dim, normalize=True)
            pool = TopKPooling(hidden_dim, ratio=pooling_ratio)
            
            self.convs.append(conv)
            self.pools.append(pool)
        
        # 注意力层：用于聚合不同层的特征
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # mean + max pooling
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 嵌入
        x = self.item_embedding(x.squeeze(-1))
        
        # 收集每层的全局特征
        layer_features = []
        
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            x, edge_index, _, batch, _ = self.pools[i](x, edge_index, batch=batch)
            
            # 全局特征
            global_feat = torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch)
            ], dim=1)
            
            layer_features.append(global_feat)
        
        # 使用注意力聚合层特征
        # [batch_size, num_layers, feature_dim]
        stacked_features = torch.stack(layer_features, dim=1)
        
        # 注意力聚合
        attended_features, _ = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # 取平均或最后一层
        x = attended_features.mean(dim=1)
        
        # 分类
        x = self.classifier(x)
        return torch.sigmoid(x).squeeze(-1)


class Set2SetSessionGNN(nn.Module):
    """
    使用Set2Set聚合的会话GNN模型
    
    Set2Set是一种更复杂的图级别特征聚合方法
    """
    
    def __init__(self,
                 num_items: int,
                 embed_dim: int = 128,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 processing_steps: int = 3):
        super(Set2SetSessionGNN, self).__init__()
        
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 物品嵌入
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # 图卷积层
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            conv = SAGEConv(in_dim, hidden_dim, normalize=True)
            self.convs.append(conv)
        
        # Set2Set聚合
        self.set2set = Set2Set(hidden_dim, processing_steps=processing_steps)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Set2Set输出维度是输入的2倍
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 嵌入
        x = self.item_embedding(x.squeeze(-1))
        
        # 图卷积
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Set2Set聚合
        x = self.set2set(x, batch)
        
        # 分类
        x = self.classifier(x)
        return torch.sigmoid(x).squeeze(-1)


def create_model(model_type: str, 
                 num_items: int,
                 config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    创建模型工厂函数
    
    Args:
        model_type (str): 模型类型 ('basic', 'attention', 'set2set')
        num_items (int): 物品总数
        config (Dict): 模型配置参数
        
    Returns:
        nn.Module: 创建的模型
    """
    if config is None:
        config = {}
    
    # 默认配置
    default_config = {
        'embed_dim': 128,
        'hidden_dim': 128,
        'num_layers': 3,
        'pooling_ratio': 0.8,
        'dropout': 0.5,
        'conv_type': 'SAGEConv'
    }
    
    # 合并配置
    default_config.update(config)
    
    if model_type == 'basic':
        return SessionGNN(num_items=num_items, **default_config)
    elif model_type == 'attention':
        return AttentionSessionGNN(num_items=num_items, **default_config)
    elif model_type == 'set2set':
        return Set2SetSessionGNN(num_items=num_items, **default_config)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


class ModelUtils:
    """
    模型工具类
    
    提供模型相关的实用函数
    """
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """
        计算模型参数数量
        
        Args:
            model (nn.Module): 模型
            
        Returns:
            int: 参数总数
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_model_size(model: nn.Module) -> float:
        """
        获取模型大小（MB）
        
        Args:
            model (nn.Module): 模型
            
        Returns:
            float: 模型大小（MB）
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    @staticmethod
    def init_weights(model: nn.Module, init_type: str = 'xavier'):
        """
        初始化模型权重
        
        Args:
            model (nn.Module): 模型
            init_type (str): 初始化类型 ('xavier', 'kaiming', 'normal')
        """
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(module.weight)
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                elif init_type == 'normal':
                    nn.init.normal_(module.weight, 0, 0.02)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)


def main():
    """
    主函数：演示模型创建和基本操作
    """
    print("=== RecSys Challenge 2015 模型演示 ===\n")
    
    # 模拟参数
    num_items = 1000
    batch_size = 32
    
    # 1. 创建不同类型的模型
    models = {
        'basic': create_model('basic', num_items),
        'attention': create_model('attention', num_items),
        'set2set': create_model('set2set', num_items)
    }
    
    # 2. 分析模型
    print("=== 模型分析 ===")
    for name, model in models.items():
        num_params = ModelUtils.count_parameters(model)
        model_size = ModelUtils.get_model_size(model)
        
        print(f"\n{name.upper()} 模型:")
        print(f"  参数数量: {num_params:,}")
        print(f"  模型大小: {model_size:.2f} MB")
        print(f"  模型结构:")
        print(f"    嵌入维度: {model.embed_dim}")
        print(f"    隐藏维度: {model.hidden_dim}")
        print(f"    网络层数: {model.num_layers}")
    
    # 3. 创建模拟数据并测试前向传播
    print(f"\n=== 前向传播测试 ===")
    
    # 创建模拟的图数据
    from torch_geometric.data import Data, Batch
    
    # 创建一个简单的图
    x = torch.randint(0, num_items, (10, 1))  # 10个节点
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)  # 边
    y = torch.tensor([1.0])  # 标签
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # 创建批次
    batch = Batch.from_data_list([data] * batch_size)
    
    # 测试每个模型
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            try:
                output = model(batch)
                print(f"{name} 模型输出形状: {output.shape}")
                print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
                print(f"  平均预测: {output.mean():.4f}")
            except Exception as e:
                print(f"{name} 模型测试失败: {e}")
    
    print("\n模型创建和测试完成！")


if __name__ == "__main__":
    main()