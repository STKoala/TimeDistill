"""
配置加载器 - 支持从YAML文件加载配置，并支持命令行参数覆盖
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from .config import Chronos2DistillConfig, ModelConfig, DataConfig, TrainingConfig, LossConfig, SystemConfig


def load_config_from_yaml(yaml_path: str) -> Chronos2DistillConfig:
    """
    从YAML文件加载配置
    
    Parameters
    ----------
    yaml_path : str
        YAML配置文件路径
        
    Returns
    -------
    Chronos2DistillConfig
        配置对象
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 如果YAML是空文件或None，使用默认配置
    if config_dict is None:
        config_dict = {}
    
    return Chronos2DistillConfig(**config_dict)


def merge_config_with_args(config: Chronos2DistillConfig, args: argparse.Namespace) -> Chronos2DistillConfig:
    """
    用命令行参数覆盖配置
    
    Parameters
    ----------
    config : Chronos2DistillConfig
        基础配置对象
    args : argparse.Namespace
        命令行参数
        
    Returns
    -------
    Chronos2DistillConfig
        合并后的配置对象
    """
    # 模型配置
    if hasattr(args, 'teacher_model_id') and args.teacher_model_id is not None:
        config.model.teacher_model_id = args.teacher_model_id
    
    # 数据配置
    if hasattr(args, 'data_dir') and args.data_dir is not None:
        config.data.data_dir = args.data_dir
    if hasattr(args, 'dataset_names') and args.dataset_names is not None:
        config.data.dataset_names = args.dataset_names
    if hasattr(args, 'context_length') and args.context_length is not None:
        config.data.context_length = args.context_length
    if hasattr(args, 'horizon') and args.horizon is not None:
        config.data.horizon = args.horizon
    if hasattr(args, 'stride') and args.stride is not None:
        config.data.stride = args.stride
    
    # 训练配置
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if hasattr(args, 'num_epochs') and args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    if hasattr(args, 'max_grad_norm') and args.max_grad_norm is not None:
        config.training.max_grad_norm = args.max_grad_norm
    
    # 损失配置
    if hasattr(args, 'alpha_pred_distill') and args.alpha_pred_distill is not None:
        config.loss.alpha_pred_distill = args.alpha_pred_distill
    if hasattr(args, 'beta_feature') and args.beta_feature is not None:
        config.loss.beta_feature = args.beta_feature
    if hasattr(args, 'dtw_weight') and args.dtw_weight is not None:
        config.loss.dtw_weight = args.dtw_weight
    if hasattr(args, 'dtw_gamma') and args.dtw_gamma is not None:
        config.loss.dtw_gamma = args.dtw_gamma
    
    # 系统配置
    if hasattr(args, 'device') and args.device is not None:
        config.system.device = args.device
    if hasattr(args, 'output_dir') and args.output_dir is not None:
        config.system.output_dir = args.output_dir
    if hasattr(args, 'verbose') and args.verbose is not None:
        config.system.verbose = args.verbose
    
    return config


def create_arg_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器
    
    Returns
    -------
    argparse.ArgumentParser
        参数解析器
    """
    parser = argparse.ArgumentParser(description='Chronos-2 知识蒸馏训练')
    
    # 配置文件
    parser.add_argument('--config', type=str, default=None,
                       help='YAML配置文件路径')
    
    # 模型配置
    parser.add_argument('--teacher_model_id', type=str, default=None,
                       help='教师模型ID')
    
    # 数据配置
    parser.add_argument('--data_dir', type=str, default=None,
                       help='数据目录路径')
    parser.add_argument('--dataset_names', type=str, nargs='+', default=None,
                       help='数据集名称列表')
    parser.add_argument('--context_length', type=int, default=None,
                       help='上下文长度')
    parser.add_argument('--horizon', type=int, default=None,
                       help='预测长度')
    parser.add_argument('--stride', type=int, default=None,
                       help='滑动窗口步长')
    
    # 训练配置
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='学习率')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--max_grad_norm', type=float, default=None,
                       help='梯度裁剪阈值')
    
    # 损失配置
    parser.add_argument('--alpha_pred_distill', type=float, default=None,
                       help='预测蒸馏权重')
    parser.add_argument('--beta_feature', type=float, default=None,
                       help='特征蒸馏权重')
    parser.add_argument('--dtw_weight', type=float, default=None,
                       help='DTW损失权重')
    parser.add_argument('--dtw_gamma', type=float, default=None,
                       help='DTW损失gamma参数')
    
    # 系统配置
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (cuda:0, cpu等)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--verbose', action='store_true',
                       help='是否打印详细信息')
    
    return parser


def load_config(config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> Chronos2DistillConfig:
    """
    加载配置的主函数
    
    Parameters
    ----------
    config_path : Optional[str]
        YAML配置文件路径，如果为None则使用默认配置
    args : Optional[argparse.Namespace]
        命令行参数，用于覆盖配置
        
    Returns
    -------
    Chronos2DistillConfig
        配置对象
    """
    # 从YAML加载或使用默认配置
    if config_path and os.path.exists(config_path):
        config = load_config_from_yaml(config_path)
    else:
        config = Chronos2DistillConfig()
    
    # 用命令行参数覆盖
    if args:
        config = merge_config_with_args(config, args)
    
    return config

