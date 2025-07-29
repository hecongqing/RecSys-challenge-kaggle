"""
RecSys Challenge 2015 数据预处理模块

这个模块负责：
1. 下载RecSys Challenge 2015数据集
2. 数据清洗和预处理
3. 特征工程
4. 数据集划分

作者: GNN Tutorial
日期: 2024
"""

import pandas as pd
import numpy as np
import os
import requests
from typing import Tuple, Optional
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class RecSysDataPreprocessor:
    """
    RecSys Challenge 2015 数据预处理类
    
    主要功能：
    - 下载和加载原始数据
    - 数据清洗和预处理
    - 创建会话图结构
    - 生成训练/验证/测试集
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据预处理器
        
        Args:
            data_dir (str): 数据存储目录
        """
        self.data_dir = data_dir
        self.clicks_file = os.path.join(data_dir, "yoochoose-clicks.dat")
        self.buys_file = os.path.join(data_dir, "yoochoose-buys.dat")
        
        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        # 数据变量
        self.df_clicks = None
        self.df_buys = None
        self.item_encoder = LabelEncoder()
        
    def download_data(self, force_download: bool = False) -> None:
        """
        下载RecSys Challenge 2015数据集
        
        Args:
            force_download (bool): 是否强制重新下载
            
        注意：由于原始数据集很大(几GB)，这里提供下载提示
        """
        print("=== RecSys Challenge 2015 数据下载指南 ===")
        print("数据集下载地址: https://www.kaggle.com/competitions/recsys-challenge-2015/data")
        print("或者: https://2015.recsyschallenge.com/challenge.html")
        print()
        print("请下载以下文件到 data/ 目录:")
        print("1. yoochoose-clicks.dat (点击数据)")
        print("2. yoochoose-buys.dat (购买数据)")
        print()
        
        # 检查文件是否存在
        if os.path.exists(self.clicks_file) and os.path.exists(self.buys_file):
            print("✓ 数据文件已存在!")
            return
        else:
            print("✗ 数据文件不存在，请手动下载")
            print(f"预期文件位置:")
            print(f"  - {self.clicks_file}")
            print(f"  - {self.buys_file}")
            
    def load_raw_data(self, sample_size: Optional[int] = 1000000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载原始数据
        
        Args:
            sample_size (int, optional): 采样大小，None表示加载全部数据
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 点击数据和购买数据
        """
        print("正在加载原始数据...")
        
        # 检查文件是否存在
        if not os.path.exists(self.clicks_file):
            print(f"错误: 找不到点击数据文件 {self.clicks_file}")
            print("请先运行 download_data() 获取数据")
            return None, None
            
        if not os.path.exists(self.buys_file):
            print(f"错误: 找不到购买数据文件 {self.buys_file}")
            print("请先运行 download_data() 获取数据")
            return None, None
        
        # 加载点击数据
        print("加载点击数据...")
        df_clicks = pd.read_csv(
            self.clicks_file, 
            header=None, 
            names=['session_id', 'timestamp', 'item_id', 'category']
        )
        
        # 加载购买数据
        print("加载购买数据...")
        df_buys = pd.read_csv(
            self.buys_file,
            header=None,
            names=['session_id', 'timestamp', 'item_id', 'price', 'quantity']
        )
        
        print(f"原始点击数据形状: {df_clicks.shape}")
        print(f"原始购买数据形状: {df_buys.shape}")
        
        # 数据采样（用于演示和快速实验）
        if sample_size is not None and sample_size < len(df_clicks['session_id'].unique()):
            print(f"正在采样 {sample_size} 个session...")
            sampled_sessions = np.random.choice(
                df_clicks['session_id'].unique(), 
                sample_size, 
                replace=False
            )
            df_clicks = df_clicks[df_clicks['session_id'].isin(sampled_sessions)]
            df_buys = df_buys[df_buys['session_id'].isin(sampled_sessions)]
            
            print(f"采样后点击数据形状: {df_clicks.shape}")
            print(f"采样后购买数据形状: {df_buys.shape}")
        
        self.df_clicks = df_clicks
        self.df_buys = df_buys
        
        return df_clicks, df_buys
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        预处理数据
        
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        if self.df_clicks is None:
            raise ValueError("请先调用 load_raw_data() 加载数据")
            
        print("开始数据预处理...")
        df = self.df_clicks.copy()
        
        # 1. 数据清洗
        print("1. 数据清洗...")
        
        # 去除重复记录
        original_len = len(df)
        df = df.drop_duplicates()
        print(f"   去除 {original_len - len(df)} 条重复记录")
        
        # 去除缺失值
        df = df.dropna()
        print(f"   最终数据形状: {df.shape}")
        
        # 2. 时间戳处理
        print("2. 时间戳处理...")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['session_id', 'timestamp'])
        
        # 3. 商品ID编码
        print("3. 商品ID编码...")
        df['item_id'] = self.item_encoder.fit_transform(df['item_id'])
        
        # 4. 添加购买标签
        print("4. 添加购买标签...")
        df['label'] = df['session_id'].isin(self.df_buys['session_id'])
        
        # 5. 会话统计信息
        print("5. 生成会话统计信息...")
        session_stats = df.groupby('session_id').agg({
            'item_id': 'count',
            'timestamp': ['min', 'max'],
            'label': 'first'
        }).round(2)
        
        session_stats.columns = ['session_length', 'start_time', 'end_time', 'has_purchase']
        session_stats['session_duration'] = (
            session_stats['end_time'] - session_stats['start_time']
        ).dt.total_seconds()
        
        print(f"会话统计信息:")
        print(f"   总会话数: {len(session_stats)}")
        print(f"   平均会话长度: {session_stats['session_length'].mean():.2f}")
        print(f"   购买转化率: {session_stats['has_purchase'].mean():.4f}")
        
        # 6. 过滤短会话（长度小于2的会话无法构建边）
        valid_sessions = session_stats[session_stats['session_length'] >= 2].index
        df = df[df['session_id'].isin(valid_sessions)]
        print(f"   过滤后有效会话数: {len(valid_sessions)}")
        
        return df
    
    def create_session_sequences(self, df: pd.DataFrame) -> dict:
        """
        为每个会话创建物品序列
        
        Args:
            df (pd.DataFrame): 预处理后的数据
            
        Returns:
            dict: 会话序列字典
        """
        print("创建会话序列...")
        
        sessions = {}
        for session_id, group in tqdm(df.groupby('session_id'), desc="处理会话"):
            # 按时间排序
            group = group.sort_values('timestamp')
            
            sessions[session_id] = {
                'items': group['item_id'].tolist(),
                'timestamps': group['timestamp'].tolist(),
                'categories': group['category'].tolist(),
                'label': group['label'].iloc[0]
            }
        
        print(f"创建了 {len(sessions)} 个会话序列")
        return sessions
    
    def split_dataset(self, df: pd.DataFrame, 
                     train_ratio: float = 0.8, 
                     val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        划分训练/验证/测试集
        
        Args:
            df (pd.DataFrame): 预处理后的数据
            train_ratio (float): 训练集比例
            val_ratio (float): 验证集比例
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 训练/验证/测试集
        """
        print("划分数据集...")
        
        # 获取所有唯一的session_id
        unique_sessions = df['session_id'].unique()
        n_sessions = len(unique_sessions)
        
        # 随机打乱
        np.random.shuffle(unique_sessions)
        
        # 计算划分点
        train_end = int(n_sessions * train_ratio)
        val_end = int(n_sessions * (train_ratio + val_ratio))
        
        # 划分session
        train_sessions = unique_sessions[:train_end]
        val_sessions = unique_sessions[train_end:val_end]
        test_sessions = unique_sessions[val_end:]
        
        # 根据session划分数据
        train_df = df[df['session_id'].isin(train_sessions)]
        val_df = df[df['session_id'].isin(val_sessions)]
        test_df = df[df['session_id'].isin(test_sessions)]
        
        print(f"数据集划分完成:")
        print(f"   训练集: {len(train_sessions)} 会话, {len(train_df)} 条记录")
        print(f"   验证集: {len(val_sessions)} 会话, {len(val_df)} 条记录")
        print(f"   测试集: {len(test_sessions)} 会话, {len(test_df)} 条记录")
        
        return train_df, val_df, test_df
    
    def get_data_statistics(self, df: pd.DataFrame) -> dict:
        """
        获取数据统计信息
        
        Args:
            df (pd.DataFrame): 数据框
            
        Returns:
            dict: 统计信息字典
        """
        stats = {
            'total_sessions': df['session_id'].nunique(),
            'total_items': df['item_id'].nunique(),
            'total_interactions': len(df),
            'avg_session_length': df.groupby('session_id').size().mean(),
            'purchase_rate': df.groupby('session_id')['label'].first().mean(),
            'unique_categories': df['category'].nunique()
        }
        
        return stats


def main():
    """
    主函数：演示数据预处理流程
    """
    print("=== RecSys Challenge 2015 数据预处理演示 ===\n")
    
    # 1. 初始化预处理器
    preprocessor = RecSysDataPreprocessor()
    
    # 2. 下载数据（显示下载指南）
    preprocessor.download_data()
    
    # 注意：如果没有真实数据，这里会创建一些模拟数据用于演示
    if not os.path.exists(preprocessor.clicks_file):
        print("\n由于没有真实数据，创建模拟数据用于演示...")
        create_demo_data(preprocessor.data_dir)
    
    # 3. 加载和预处理数据
    df_clicks, df_buys = preprocessor.load_raw_data(sample_size=10000)  # 小样本演示
    
    if df_clicks is not None:
        df_processed = preprocessor.preprocess_data()
        
        # 4. 数据集划分
        train_df, val_df, test_df = preprocessor.split_dataset(df_processed)
        
        # 5. 统计信息
        print(f"\n=== 最终数据统计 ===")
        for split_name, split_df in [("训练集", train_df), ("验证集", val_df), ("测试集", test_df)]:
            stats = preprocessor.get_data_statistics(split_df)
            print(f"\n{split_name}:")
            for key, value in stats.items():
                print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")


def create_demo_data(data_dir: str):
    """
    创建演示用的模拟数据
    
    Args:
        data_dir (str): 数据目录
    """
    print("创建模拟数据...")
    
    # 模拟点击数据
    np.random.seed(42)
    n_sessions = 1000
    n_items = 100
    
    clicks_data = []
    buys_data = []
    
    for session_id in tqdm(range(1, n_sessions + 1), desc="生成模拟会话"):
        # 随机会话长度
        session_length = np.random.randint(2, 20)
        
        # 随机选择物品
        items = np.random.choice(n_items, session_length, replace=False)
        
        # 生成时间戳
        base_time = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 30))
        
        for i, item_id in enumerate(items):
            timestamp = base_time + pd.Timedelta(minutes=i * np.random.randint(1, 10))
            category = f"category_{item_id % 10}"
            
            clicks_data.append([session_id, timestamp, item_id, category])
        
        # 随机决定是否购买（10%概率）
        if np.random.random() < 0.1:
            # 购买最后一个物品
            buy_timestamp = timestamp + pd.Timedelta(minutes=np.random.randint(1, 5))
            price = np.random.uniform(10, 100)
            quantity = np.random.randint(1, 3)
            
            buys_data.append([session_id, buy_timestamp, items[-1], price, quantity])
    
    # 保存数据
    clicks_df = pd.DataFrame(clicks_data, columns=['session_id', 'timestamp', 'item_id', 'category'])
    buys_df = pd.DataFrame(buys_data, columns=['session_id', 'timestamp', 'item_id', 'price', 'quantity'])
    
    clicks_df.to_csv(os.path.join(data_dir, 'yoochoose-clicks.dat'), header=False, index=False)
    buys_df.to_csv(os.path.join(data_dir, 'yoochoose-buys.dat'), header=False, index=False)
    
    print(f"模拟数据已保存到 {data_dir}")
    print(f"点击数据: {len(clicks_df)} 条记录")
    print(f"购买数据: {len(buys_df)} 条记录")


if __name__ == "__main__":
    main()