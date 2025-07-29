"""
RecSys Challenge 2015 PyTorch Geometric数据集模块

这个模块负责：
1. 将会话数据转换为图结构
2. 创建PyTorch Geometric数据集
3. 数据加载和批处理

每个会话被表示为一个图：
- 节点：会话中的物品
- 边：物品之间的时序关系（前一个物品指向后一个物品）
- 节点特征：物品的嵌入表示
- 图标签：该会话是否产生购买行为

作者: GNN Tutorial
日期: 2024
"""

import torch
from torch_geometric.data import InMemoryDataset, Data
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pickle
from typing import List, Dict, Tuple, Optional


class YooChooseDataset(InMemoryDataset):
    """
    RecSys Challenge 2015 数据集类
    
    将每个用户会话转换为一个图：
    - 每个物品作为图中的一个节点
    - 连续的物品之间有边连接（时序关系）
    - 图的标签表示该会话是否产生购买行为
    """
    
    def __init__(self, 
                 root: str, 
                 df: pd.DataFrame,
                 name: str = "yoochoose",
                 max_item_id: Optional[int] = None,
                 transform=None, 
                 pre_transform=None):
        """
        初始化数据集
        
        Args:
            root (str): 数据集根目录
            df (pd.DataFrame): 预处理后的数据框
            name (str): 数据集名称
            max_item_id (Optional[int]): 最大允许的item_id值，用于边界检查
            transform: 数据变换函数
            pre_transform: 预变换函数
        """
        self.df = df
        self.name = name
        self._max_item_id = max_item_id
        
        # 为节点特征创建物品编码器
        self.item_encoder = LabelEncoder()
        
        super(YooChooseDataset, self).__init__(root, transform, pre_transform)
        
        # 加载处理后的数据
        # 修复PyTorch 2.6的torch.load安全性更新问题
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        
    @property
    def raw_file_names(self) -> List[str]:
        """原始文件名列表"""
        return []
    
    @property
    def processed_file_names(self) -> List[str]:
        """处理后文件名列表"""
        return [f'{self.name}_processed.pt']
    
    def download(self):
        """下载数据（这里不需要，因为数据已经预处理过）"""
        pass
    
    def process(self):
        """
        处理数据：将每个会话转换为图结构
        
        处理步骤：
        1. 按session_id分组
        2. 为每个会话创建图
        3. 建立节点和边
        4. 设置节点特征和图标签
        """
        print("开始处理数据，创建图结构...")
        
        data_list = []
        
        # 按会话分组处理
        grouped = self.df.groupby('session_id')
        
        for session_id, group in tqdm(grouped, desc="处理会话图"):
            # 按时间排序确保正确的时序关系
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            # 为当前会话的物品重新编码（从0开始）
            # 这是必需的，因为图中的节点索引必须从0开始且连续
            session_item_encoder = LabelEncoder()
            group['node_id'] = session_item_encoder.fit_transform(group['item_id'])
            
            # 创建节点特征矩阵
            # 每个节点的特征是原始的item_id（用于后续的嵌入查找）
            unique_items = group.drop_duplicates('node_id').sort_values('node_id')
            node_features = unique_items['item_id'].values
            
            # 添加边界检查：确保所有item_id都在有效范围内
            max_item_id = np.max(node_features)
            if hasattr(self, '_max_item_id'):
                if max_item_id > self._max_item_id:
                    print(f"警告: 发现超出范围的item_id {max_item_id}, 最大允许值: {self._max_item_id}")
                    # 将超出范围的item_id截断到最大值
                    node_features = np.clip(node_features, 0, self._max_item_id)
            
            x = torch.LongTensor(node_features).unsqueeze(1)  # [num_nodes, 1]
            
            # 创建边：连接连续的物品
            # 边的方向：前一个物品 -> 后一个物品
            if len(group) > 1:
                source_nodes = group['node_id'].values[:-1]  # 前n-1个节点
                target_nodes = group['node_id'].values[1:]   # 后n-1个节点
                edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            else:
                # 如果只有一个物品，创建自循环
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            
            # 图标签：是否产生购买行为
            y = torch.FloatTensor([group['label'].iloc[0]])
            
            # 创建图数据对象
            data = Data(x=x, edge_index=edge_index, y=y)
            
            # 添加额外信息（可选）
            data.session_id = session_id
            data.num_interactions = len(group)
            
            data_list.append(data)
        
        print(f"成功创建 {len(data_list)} 个图")
        
        # 保存处理后的数据
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        # 保存元数据
        metadata = {
            'num_graphs': len(data_list),
            'total_items': self.df['item_id'].nunique(),
            'total_sessions': self.df['session_id'].nunique(),
            'purchase_rate': self.df.groupby('session_id')['label'].first().mean()
        }
        
        with open(os.path.join(self.processed_dir, f'{self.name}_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print("数据处理完成！")
        print(f"元数据: {metadata}")


class SessionGraphBuilder:
    """
    会话图构建工具类
    
    提供多种图构建策略和工具函数
    """
    
    @staticmethod
    def build_sequential_graph(items: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建顺序图：每个物品只连接到下一个物品
        
        Args:
            items (List[int]): 物品序列
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 节点特征和边索引
        """
        if len(items) == 0:
            return torch.empty((0, 1), dtype=torch.long), torch.empty((2, 0), dtype=torch.long)
        
        # 去重并保持顺序
        unique_items = []
        seen = set()
        for item in items:
            if item not in seen:
                unique_items.append(item)
                seen.add(item)
        
        # 创建物品到节点的映射
        item_to_node = {item: i for i, item in enumerate(unique_items)}
        
        # 节点特征
        x = torch.LongTensor(unique_items).unsqueeze(1)
        
        # 创建边
        edges = []
        for i in range(len(items) - 1):
            source = item_to_node[items[i]]
            target = item_to_node[items[i + 1]]
            edges.append([source, target])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            # 如果没有边，创建空的边索引
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return x, edge_index
    
    @staticmethod
    def build_fully_connected_graph(items: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建全连接图：每个物品连接到所有其他物品
        
        Args:
            items (List[int]): 物品序列
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 节点特征和边索引
        """
        unique_items = list(set(items))
        n_items = len(unique_items)
        
        if n_items == 0:
            return torch.empty((0, 1), dtype=torch.long), torch.empty((2, 0), dtype=torch.long)
        
        # 节点特征
        x = torch.LongTensor(unique_items).unsqueeze(1)
        
        # 创建全连接边（包括自循环）
        edges = []
        for i in range(n_items):
            for j in range(n_items):
                edges.append([i, j])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        return x, edge_index
    
    @staticmethod
    def build_window_graph(items: List[int], window_size: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建窗口图：每个物品连接到窗口内的其他物品
        
        Args:
            items (List[int]): 物品序列
            window_size (int): 窗口大小
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 节点特征和边索引
        """
        if len(items) == 0:
            return torch.empty((0, 1), dtype=torch.long), torch.empty((2, 0), dtype=torch.long)
        
        # 去重并保持顺序
        unique_items = []
        item_positions = {}  # 记录每个物品的所有出现位置
        
        for pos, item in enumerate(items):
            if item not in item_positions:
                unique_items.append(item)
                item_positions[item] = []
            item_positions[item].append(pos)
        
        # 创建物品到节点的映射
        item_to_node = {item: i for i, item in enumerate(unique_items)}
        
        # 节点特征
        x = torch.LongTensor(unique_items).unsqueeze(1)
        
        # 创建窗口内的边
        edges = set()
        for i, item_i in enumerate(items):
            for j in range(max(0, i - window_size), min(len(items), i + window_size + 1)):
                if i != j:
                    item_j = items[j]
                    source = item_to_node[item_i]
                    target = item_to_node[item_j]
                    edges.add((source, target))
        
        if edges:
            edge_index = torch.tensor(list(edges), dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return x, edge_index


def create_datasets(train_df: pd.DataFrame, 
                   val_df: pd.DataFrame, 
                   test_df: pd.DataFrame,
                   root_dir: str = "data/processed") -> Tuple[YooChooseDataset, YooChooseDataset, YooChooseDataset]:
    """
    创建训练、验证和测试数据集
    
    Args:
        train_df (pd.DataFrame): 训练数据
        val_df (pd.DataFrame): 验证数据
        test_df (pd.DataFrame): 测试数据
        root_dir (str): 根目录
        
    Returns:
        Tuple[YooChooseDataset, YooChooseDataset, YooChooseDataset]: 三个数据集
    """
    print("创建PyTorch Geometric数据集...")
    
    # 确保目录存在
    os.makedirs(root_dir, exist_ok=True)
    
    # 计算所有数据中的最大item_id，用于边界检查
    all_dfs = [train_df, val_df, test_df]
    max_item_id = max(df['item_id'].max() for df in all_dfs if len(df) > 0)
    print(f"数据中最大item_id: {max_item_id}")
    
    # 创建数据集
    train_dataset = YooChooseDataset(
        root=os.path.join(root_dir, "train"),
        df=train_df,
        name="train",
        max_item_id=max_item_id
    )
    
    val_dataset = YooChooseDataset(
        root=os.path.join(root_dir, "val"),
        df=val_df,
        name="val",
        max_item_id=max_item_id
    )
    
    test_dataset = YooChooseDataset(
        root=os.path.join(root_dir, "test"),
        df=test_df,
        name="test",
        max_item_id=max_item_id
    )
    
    print(f"数据集创建完成:")
    print(f"  训练集: {len(train_dataset)} 个图")
    print(f"  验证集: {len(val_dataset)} 个图")
    print(f"  测试集: {len(test_dataset)} 个图")
    
    return train_dataset, val_dataset, test_dataset


def analyze_dataset(dataset: YooChooseDataset) -> Dict:
    """
    分析数据集统计信息
    
    Args:
        dataset (YooChooseDataset): 数据集
        
    Returns:
        Dict: 统计信息字典
    """
    print(f"分析数据集 ({len(dataset)} 个图)...")
    
    num_nodes_list = []
    num_edges_list = []
    labels = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        num_nodes_list.append(data.num_nodes)
        num_edges_list.append(data.num_edges)
        labels.append(data.y.item())
    
    stats = {
        'num_graphs': len(dataset),
        'avg_num_nodes': np.mean(num_nodes_list),
        'avg_num_edges': np.mean(num_edges_list),
        'min_nodes': np.min(num_nodes_list),
        'max_nodes': np.max(num_nodes_list),
        'min_edges': np.min(num_edges_list),
        'max_edges': np.max(num_edges_list),
        'purchase_rate': np.mean(labels),
        'total_positive': np.sum(labels),
        'total_negative': len(labels) - np.sum(labels)
    }
    
    return stats


def main():
    """
    主函数：演示数据集创建和分析
    """
    print("=== RecSys Challenge 2015 数据集创建演示 ===\n")
    
    # 这里需要先运行数据预处理
    from src.data_preprocessing import RecSysDataPreprocessor
    
    # 1. 数据预处理
    preprocessor = RecSysDataPreprocessor()
    
    # 检查是否有数据文件，如果没有则创建演示数据
    if not os.path.exists(preprocessor.clicks_file):
        from src.data_preprocessing import create_demo_data
        create_demo_data(preprocessor.data_dir)
    
    # 加载数据
    df_clicks, df_buys = preprocessor.load_raw_data(sample_size=1000)  # 小样本演示
    if df_clicks is not None:
        df_processed = preprocessor.preprocess_data()
        train_df, val_df, test_df = preprocessor.split_dataset(df_processed)
        
        # 2. 创建数据集
        train_dataset, val_dataset, test_dataset = create_datasets(
            train_df, val_df, test_df
        )
        
        # 3. 分析数据集
        print("\n=== 数据集统计信息 ===")
        for name, dataset in [("训练集", train_dataset), ("验证集", val_dataset), ("测试集", test_dataset)]:
            print(f"\n{name}:")
            stats = analyze_dataset(dataset)
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        # 4. 示例：查看第一个图
        print("\n=== 示例图结构 ===")
        example_graph = train_dataset[0]
        print(f"节点数: {example_graph.num_nodes}")
        print(f"边数: {example_graph.num_edges}")
        print(f"节点特征形状: {example_graph.x.shape}")
        print(f"边索引形状: {example_graph.edge_index.shape}")
        print(f"标签: {example_graph.y.item()}")
        print(f"会话ID: {example_graph.session_id}")


if __name__ == "__main__":
    main()