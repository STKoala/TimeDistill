"""
Chronos-2 蒸馏脚本
使用 Hugging Face TRL 的 GKDTrainer 进行知识蒸馏

注意：由于 Chronos-2 使用自定义架构和时间序列 tokenization，
本脚本提供了一个适配方案，将时间序列数据转换为适合蒸馏的格式。
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer
from trl.experimental.gkd import GKDConfig, GKDTrainer
from chronos import Chronos2Pipeline, Chronos2Model
from chronos.chronos2 import Chronos2ForecastingConfig, Chronos2CoreConfig
import warnings
warnings.filterwarnings('ignore')


def prepare_time_series_data(
    data_paths: List[str],
    context_length: int = 96,
    horizon: int = 24,
    stride: int = 1,
    target_col: str = "OT"
) -> List[Dict[str, Any]]:
    """
    准备时间序列数据用于蒸馏
    
    Parameters
    ----------
    data_paths : List[str]
        数据文件路径列表
    context_length : int
        上下文长度
    horizon : int
        预测长度
    stride : int
        滑动窗口步长
    target_col : str
        目标列名
        
    Returns
    -------
    List[Dict[str, Any]]
        格式化的数据列表，每个元素包含 target, past_covariates, future_covariates
    """
    all_samples = []
    
    for data_path in data_paths:
        if not os.path.exists(data_path):
            print(f"警告: 未找到数据文件 {data_path}，跳过")
            continue
            
        df = pd.read_csv(data_path)
        
        # 确定目标列
        if target_col not in df.columns:
            target_col = df.columns[-1]
            print(f"{data_path}: 未找到 {target_col} 列，使用最后一列: {target_col}")
        
        # 提取时间序列
        ts_values = df[target_col].values.astype(np.float32)
        
        # 使用滑动窗口创建样本
        max_start_idx = len(ts_values) - context_length - horizon
        
        for start_idx in range(0, max_start_idx + 1, stride):
            context = ts_values[start_idx:start_idx + context_length]
            target = ts_values[start_idx + context_length:start_idx + context_length + horizon]
            
            # 转换为 Chronos-2 需要的格式
            sample = {
                "target": torch.tensor(context, dtype=torch.float32),
                "past_covariates": {},
                "future_covariates": {}
            }
            all_samples.append(sample)
    
    print(f"共准备 {len(all_samples)} 个训练样本")
    return all_samples


def create_student_model(
    teacher_pipeline: Chronos2Pipeline,
    student_config_overrides: Optional[Dict[str, Any]] = None
) -> Chronos2Model:
    """
    创建学生模型（更小的架构）
    
    Parameters
    ----------
    teacher_pipeline : Chronos2Pipeline
        教师模型 pipeline
    student_config_overrides : Optional[Dict[str, Any]]
        学生模型配置覆盖参数，例如 {"num_layers": 3, "d_model": 384}
        
    Returns
    -------
    Chronos2Model
        学生模型
    """
    teacher_config = teacher_pipeline.model.config
    
    # 创建学生模型配置
    student_config = AutoConfig.from_pretrained(
        teacher_config.name_or_path if hasattr(teacher_config, 'name_or_path') else "amazon/chronos-2-base"
    )
    
    # 应用配置覆盖
    if student_config_overrides:
        for key, value in student_config_overrides.items():
            if hasattr(student_config, key):
                setattr(student_config, key, value)
            elif hasattr(student_config, 'chronos_config'):
                if hasattr(student_config.chronos_config, key):
                    setattr(student_config.chronos_config, key, value)
    
    # 创建学生模型
    student_model = Chronos2Model(student_config)
    
    # 可选：从教师模型初始化部分权重（例如 embedding 层）
    # 这里可以根据需要实现权重初始化策略
    
    return student_model


def convert_chronos_data_to_messages(
    chronos_samples: List[Dict[str, Any]],
    pipeline: Chronos2Pipeline
) -> List[Dict[str, Any]]:
    """
    将 Chronos-2 格式的数据转换为适合 GKD 训练的 messages 格式
    
    注意：这是一个适配层，因为 Chronos-2 使用自定义的 tokenization。
    实际实现可能需要根据 Chronos-2 的内部 tokenization 逻辑进行调整。
    
    Parameters
    ----------
    chronos_samples : List[Dict[str, Any]]
        Chronos-2 格式的样本
    pipeline : Chronos2Pipeline
        Chronos-2 pipeline，用于获取 tokenization 信息
        
    Returns
    -------
    List[Dict[str, Any]]
        转换为 messages 格式的数据
    """
    messages = []
    
    for sample in chronos_samples:
        # 将时间序列数据转换为文本格式（简化版本）
        # 实际实现中，应该使用 Chronos-2 的 tokenizer 将数值转换为 token
        
        context = sample["target"].numpy()
        target = sample.get("future_target", None)
        
        # 将数值序列转换为字符串表示（简化处理）
        # 实际应该使用 Chronos-2 的量化方法
        context_str = " ".join([f"{x:.4f}" for x in context[:50]])  # 限制长度
        if target is not None:
            target_str = " ".join([f"{x:.4f}" for x in target[:20]])
        else:
            target_str = ""
        
        message = {
            "messages": [
                {
                    "role": "user",
                    "content": f"预测时间序列: {context_str}"
                },
                {
                    "role": "assistant",
                    "content": target_str if target_str else "预测结果"
                }
            ]
        }
        messages.append(message)
    
    return messages


def main():
    """主函数：执行 Chronos-2 蒸馏"""
    
    # ========== 配置参数 ==========
    teacher_model_id = "amazon/chronos-2-base"  # 或 "amazon/chronos-2"
    output_dir = "./chronos-2-distilled_long_forecast"
    
    # 数据配置
    data_paths = [
        "datasets/ETTh1.csv",
        "datasets/ETTh2.csv",
        "datasets/ETTm1.csv",
        "datasets/ETTm2.csv"
    ]
    context_length = 720
    horizon = 96
    stride = 1
    target_col = "OT"
    
    # 学生模型配置（更小的架构）
    student_config_overrides = {
        "num_layers": 3,      # 减少层数（教师可能是 6 或 12 层）
        "d_model": 384,       # 减少隐藏层维度（教师可能是 512 或 768）
        "d_ff": 1536,         # 减少前馈网络维度
        "num_heads": 6,       # 减少注意力头数
    }
    
    # 训练配置
    training_config = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "learning_rate": 5e-5,
        "lmbda": 0.5,          # 0.0 为完全 Offline 蒸馏，1.0 为完全 On-policy
        "beta": 0.5,            # Jensen-Shannon 散度插值系数
        "temperature": 2.0,     # Logits 平滑温度
        "max_new_tokens": 128,  # 最大生成 token 数
        "max_steps": 1000,
        "save_steps": 500,
        "logging_steps": 10,
        "eval_steps": 500,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    
    # ========== 加载教师模型 ==========
    print("=" * 50)
    print("加载教师模型 (Chronos-2)...")
    print("=" * 50)
    teacher_pipeline = Chronos2Pipeline.from_pretrained(
        teacher_model_id,
        device_map="auto"
    )
    print(f"教师模型加载完成: {teacher_model_id}")
    print(f"模型参数量: {sum(p.numel() for p in teacher_pipeline.model.parameters()):,}")
    
    # ========== 创建学生模型 ==========
    print("\n" + "=" * 50)
    print("创建学生模型...")
    print("=" * 50)
    student_model = create_student_model(
        teacher_pipeline,
        student_config_overrides
    )
    student_model = student_model.to(teacher_pipeline.model.device)
    print(f"学生模型参数量: {sum(p.numel() for p in student_model.parameters()):,}")
    print(f"压缩比: {sum(p.numel() for p in student_model.parameters()) / sum(p.numel() for p in teacher_pipeline.model.parameters()):.2%}")
    
    # ========== 准备数据 ==========
    print("\n" + "=" * 50)
    print("准备训练数据...")
    print("=" * 50)
    chronos_samples = prepare_time_series_data(
        data_paths=data_paths,
        context_length=context_length,
        horizon=horizon,
        stride=stride,
        target_col=target_col
    )
    
    # 转换为 messages 格式（适配 GKD）
    # 注意：这里需要根据 Chronos-2 的实际 tokenization 进行调整
    train_messages = convert_chronos_data_to_messages(
        chronos_samples[:int(len(chronos_samples) * 0.9)],  # 90% 训练
        teacher_pipeline
    )
    eval_messages = convert_chronos_data_to_messages(
        chronos_samples[int(len(chronos_samples) * 0.9):],  # 10% 验证
        teacher_pipeline
    )
    
    train_dataset = Dataset.from_list(train_messages)
    eval_dataset = Dataset.from_list(eval_messages)
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset)}")
    
    # ========== 配置训练参数 ==========
    print("\n" + "=" * 50)
    print("配置训练参数...")
    print("=" * 50)
    training_args = GKDConfig(**training_config)
    
    # ========== 创建 Trainer ==========
    print("\n" + "=" * 50)
    print("初始化 GKDTrainer...")
    print("=" * 50)
    
    # 注意：Chronos-2 使用自定义 tokenizer，这里需要适配
    # 如果 Chronos-2 有 tokenizer，应该使用它；否则需要创建一个适配器
    try:
        # 尝试从教师模型获取 tokenizer
        if hasattr(teacher_pipeline.model, 'tokenizer'):
            tokenizer = teacher_pipeline.model.tokenizer
        else:
            # 使用 T5 tokenizer 作为基础（Chronos-2 基于 T5）
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            print("警告: 未找到 Chronos-2 tokenizer，使用 T5 tokenizer 作为替代")
    except Exception as e:
        print(f"警告: 无法加载 tokenizer: {e}")
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
    
    trainer = GKDTrainer(
        model=student_model,
        teacher_model=teacher_pipeline.model,  # 使用 inner_model
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # ========== 开始训练 ==========
    print("\n" + "=" * 50)
    print("开始蒸馏训练...")
    print("=" * 50)
    trainer.train()
    
    # ========== 保存模型 ==========
    print("\n" + "=" * 50)
    print("保存蒸馏后的模型...")
    print("=" * 50)
    trainer.save_model()
    print(f"模型已保存到: {output_dir}")
    
    print("\n" + "=" * 50)
    print("蒸馏完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()

