"""
RecSys Challenge 2015 主程序

基于图神经网络的session-based推荐系统完整实现
整合数据预处理、模型训练、评估和可视化

主要功能：
1. 数据下载和预处理
2. 图数据集创建
3. 模型训练和验证
4. 结果分析和可视化
5. 模型推理

使用方法：
python main.py --help

作者: GNN Tutorial
日期: 2024
"""

import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import os
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append('src')

# 导入自定义模块
from src.data_preprocessing import RecSysDataPreprocessor, create_demo_data
from src.dataset import create_datasets, analyze_dataset
from src.models import create_model, ModelUtils
from src.trainer import SessionTrainer, MetricsAnalyzer


def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="RecSys Challenge 2015 - 基于图神经网络的会话推荐系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='data',
                       help='数据存储目录')
    parser.add_argument('--sample_size', type=int, default=10000,
                       help='数据采样大小（None表示使用全部数据）')
    parser.add_argument('--min_session_length', type=int, default=2,
                       help='最小会话长度')
    
    # 模型相关参数
    parser.add_argument('--model_type', type=str, default='basic',
                       choices=['basic', 'attention', 'set2set'],
                       help='模型类型')
    parser.add_argument('--embed_dim', type=int, default=128,
                       help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='GNN层数')
    parser.add_argument('--pooling_ratio', type=float, default=0.8,
                       help='TopK池化比例')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout概率')
    parser.add_argument('--conv_type', type=str, default='SAGEConv',
                       choices=['SAGEConv', 'GCNConv', 'GATConv'],
                       help='图卷积类型')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停耐心值')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                       help='早停最小改善')
    
    # 实验相关参数
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--log_interval', type=int, default=50,
                       help='日志输出间隔')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'demo'],
                       help='运行模式')
    parser.add_argument('--load_model', type=str, default=None,
                       help='加载预训练模型路径')
    parser.add_argument('--create_demo_data', action='store_true',
                       help='是否创建演示数据')
    parser.add_argument('--visualize', action='store_true',
                       help='是否生成可视化图表')
    
    return parser.parse_args()


def set_random_seed(seed: int):
    """
    设置随机种子
    
    Args:
        seed (int): 随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> torch.device:
    """
    获取计算设备
    
    Args:
        device_arg (str): 设备参数
        
    Returns:
        torch.device: 计算设备
    """
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def prepare_data(args):
    """
    准备数据
    
    Args:
        args: 命令行参数
        
    Returns:
        tuple: 训练、验证、测试数据集和数据统计信息
    """
    print("=== 数据准备阶段 ===")
    
    # 1. 初始化数据预处理器
    preprocessor = RecSysDataPreprocessor(data_dir=args.data_dir)
    
    # 2. 检查数据文件
    preprocessor.download_data()
    
    # 3. 创建演示数据（如果需要）
    if args.create_demo_data or not os.path.exists(preprocessor.clicks_file):
        print("创建演示数据...")
        create_demo_data(args.data_dir)
    
    # 4. 加载和预处理数据
    df_clicks, df_buys = preprocessor.load_raw_data(sample_size=args.sample_size)
    
    if df_clicks is None:
        raise ValueError("数据加载失败，请检查数据文件")
    
    df_processed = preprocessor.preprocess_data()
    
    # 5. 数据集划分
    train_df, val_df, test_df = preprocessor.split_dataset(df_processed)
    
    # 6. 创建图数据集
    print("\n创建图数据集...")
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_df, val_df, test_df
    )
    
    # 7. 数据统计
    print("\n=== 数据统计信息 ===")
    data_stats = {}
    for name, dataset in [("训练集", train_dataset), ("验证集", val_dataset), ("测试集", test_dataset)]:
        stats = analyze_dataset(dataset)
        data_stats[name.lower()] = stats
        print(f"\n{name}:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    return train_dataset, val_dataset, test_dataset, data_stats


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size: int):
    """
    创建数据加载器
    
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        test_dataset: 测试数据集
        batch_size (int): 批量大小
        
    Returns:
        tuple: 训练、验证、测试数据加载器
    """
    print(f"\n创建数据加载器 (batch_size={batch_size})...")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    print(f"  测试批次数: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def create_and_analyze_model(args, num_items: int, device: torch.device):
    """
    创建和分析模型
    
    Args:
        args: 命令行参数
        num_items (int): 物品总数
        device (torch.device): 计算设备
        
    Returns:
        nn.Module: 创建的模型
    """
    print("\n=== 模型创建阶段 ===")
    
    # 模型配置
    model_config = {
        'embed_dim': args.embed_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'pooling_ratio': args.pooling_ratio,
        'dropout': args.dropout,
        'conv_type': args.conv_type
    }
    
    # 创建模型
    model = create_model(args.model_type, num_items, model_config)
    model = model.to(device)
    
    # 模型分析
    num_params = ModelUtils.count_parameters(model)
    model_size = ModelUtils.get_model_size(model)
    
    print(f"模型类型: {args.model_type}")
    print(f"参数数量: {num_params:,}")
    print(f"模型大小: {model_size:.2f} MB")
    print(f"模型配置: {model_config}")
    
    return model


def train_model(args, model, train_loader, val_loader, test_loader, device):
    """
    训练模型
    
    Args:
        args: 命令行参数
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        device: 计算设备
        
    Returns:
        dict: 训练结果
    """
    print("\n=== 模型训练阶段 ===")
    
    # 训练配置
    train_config = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_epochs': args.num_epochs,
        'patience': args.patience,
        'min_delta': args.min_delta,
        'save_dir': args.save_dir,
        'log_interval': args.log_interval,
        'save_best_only': True
    }
    
    # 创建训练器
    trainer = SessionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        config=train_config
    )
    
    # 开始训练
    results = trainer.train()
    
    return results, trainer


def evaluate_model(args, model, test_loader, device):
    """
    评估模型
    
    Args:
        args: 命令行参数
        model: 模型
        test_loader: 测试数据加载器
        device: 计算设备
        
    Returns:
        dict: 评估结果
    """
    print("\n=== 模型评估阶段 ===")
    
    # 创建训练器（仅用于评估）
    trainer = SessionTrainer(
        model=model,
        train_loader=test_loader,  # 占位
        val_loader=test_loader,    # 占位
        test_loader=test_loader,
        device=device
    )
    
    # 加载模型
    if args.load_model:
        trainer.load_model(args.load_model)
    
    # 评估
    test_loss, test_auc, test_metrics = trainer.validate(test_loader, "测试评估中")
    
    print(f"测试结果:")
    print(f"  损失: {test_loss:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  准确率: {test_metrics['accuracy']:.4f}")
    print(f"  精确率: {test_metrics['precision']:.4f}")
    print(f"  召回率: {test_metrics['recall']:.4f}")
    print(f"  F1分数: {test_metrics['f1']:.4f}")
    
    return test_metrics


def visualize_results(args, results=None, test_metrics=None):
    """
    可视化结果
    
    Args:
        args: 命令行参数
        results: 训练结果
        test_metrics: 测试指标
    """
    if not args.visualize:
        return
    
    print("\n=== 结果可视化阶段 ===")
    
    # 创建可视化目录
    viz_dir = os.path.join(args.save_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    if results and 'history' in results:
        # 训练历史
        MetricsAnalyzer.plot_training_history(
            results['history'],
            save_path=os.path.join(viz_dir, 'training_history.png')
        )
    
    if test_metrics and 'predictions' in test_metrics:
        # 混淆矩阵
        MetricsAnalyzer.plot_confusion_matrix(
            test_metrics['labels'],
            test_metrics['predictions'],
            save_path=os.path.join(viz_dir, 'confusion_matrix.png')
        )
        
        # 预测分布
        MetricsAnalyzer.plot_prediction_distribution(
            test_metrics['labels'],
            test_metrics['predictions'],
            save_path=os.path.join(viz_dir, 'prediction_distribution.png')
        )
        
        # 阈值分析
        threshold_analysis = MetricsAnalyzer.analyze_predictions(
            test_metrics['labels'],
            test_metrics['predictions']
        )
        
        # 保存阈值分析
        threshold_analysis.to_csv(
            os.path.join(viz_dir, 'threshold_analysis.csv'),
            index=False
        )
        print(f"阈值分析已保存: {os.path.join(viz_dir, 'threshold_analysis.csv')}")


def save_experiment_log(args, results, data_stats, save_dir):
    """
    保存实验日志
    
    Args:
        args: 命令行参数
        results: 训练结果
        data_stats: 数据统计
        save_dir: 保存目录
    """
    log_data = {
        'experiment_time': datetime.now().isoformat(),
        'args': vars(args),
        'data_stats': data_stats,
        'results': {
            k: v for k, v in results.items() 
            if k != 'history'  # history单独保存
        } if results else None
    }
    
    log_path = os.path.join(save_dir, 'experiment_log.json')
    import json
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"实验日志已保存: {log_path}")


def demo_mode(args):
    """
    演示模式：使用小数据快速演示整个流程
    
    Args:
        args: 命令行参数
    """
    print("=== 演示模式 ===")
    print("使用小数据集快速演示整个训练流程")
    
    # 强制使用小数据和少量epoch
    args.sample_size = 1000
    args.num_epochs = 5
    args.batch_size = 32
    args.create_demo_data = True
    args.visualize = True
    
    print(f"演示配置:")
    print(f"  数据样本: {args.sample_size}")
    print(f"  训练轮数: {args.num_epochs}")
    print(f"  批量大小: {args.batch_size}")
    
    # 运行完整流程
    main_pipeline(args)


def main_pipeline(args):
    """
    主要训练管道
    
    Args:
        args: 命令行参数
    """
    # 1. 数据准备
    train_dataset, val_dataset, test_dataset, data_stats = prepare_data(args)
    
    # 2. 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, args.batch_size
    )
    
    # 3. 获取物品数量和设备
    num_items = data_stats['训练集']['total_items'] + 1  # +1 for padding
    device = get_device(args.device)
    
    # 4. 创建模型
    model = create_and_analyze_model(args, num_items, device)
    
    # 5. 训练或评估
    results = None
    test_metrics = None
    
    if args.mode == 'train':
        results, trainer = train_model(args, model, train_loader, val_loader, test_loader, device)
        test_metrics = results['test_metrics']
    elif args.mode == 'eval':
        test_metrics = evaluate_model(args, model, test_loader, device)
    
    # 6. 可视化
    visualize_results(args, results, test_metrics)
    
    # 7. 保存实验日志
    save_experiment_log(args, results, data_stats, args.save_dir)
    
    print(f"\n=== 实验完成 ===")
    if results:
        print(f"最佳验证AUC: {results['best_val_auc']:.4f}")
    if test_metrics:
        print(f"最终测试AUC: {test_metrics['auc']:.4f}")
    print(f"结果保存在: {args.save_dir}")


def main():
    """
    主函数
    """
    # 解析参数
    args = parse_arguments()
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 打印开始信息
    print("=" * 60)
    print("RecSys Challenge 2015 - 基于图神经网络的会话推荐系统")
    print("=" * 60)
    print(f"运行模式: {args.mode}")
    print(f"模型类型: {args.model_type}")
    print(f"数据样本: {args.sample_size}")
    print(f"随机种子: {args.seed}")
    print("=" * 60)
    
    # 根据模式运行
    try:
        if args.mode == 'demo':
            demo_mode(args)
        else:
            main_pipeline(args)
    except KeyboardInterrupt:
        print("\n用户中断训练")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n程序结束")


if __name__ == "__main__":
    main()