#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chronos-2 蒸馏入口脚本（特征蒸馏 + DTW）

用法（示例）：
  # 使用默认配置
  python -m TimeDistill.scripts.chronos2_distill_feature_dtw
  
  # 使用YAML配置文件
  python -m TimeDistill.scripts.chronos2_distill_feature_dtw --config configs/chronos-2-distill.yaml
  
  # 使用YAML配置文件并覆盖部分参数
  python -m TimeDistill.scripts.chronos2_distill_feature_dtw --config configs/chronos-2-distill.yaml --batch_size 128 --learning_rate 2e-5

说明：
- 该脚本仅负责"组装配置/创建数据集与DataLoader/创建trainer并启动训练"
- 训练主逻辑在 `TimeDistill.trainers.Chronos2DistillationTrainer`
- 支持从YAML配置文件加载配置，并支持命令行参数覆盖
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import asdict

import torch
from chronos import Chronos2Pipeline
from torch.utils.data import DataLoader

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from TimeDistill.configs import load_config, create_arg_parser
from TimeDistill.datasets import Chronos2DistillationDataset
from TimeDistill.models import create_student_model
from TimeDistill.trainers import Chronos2DistillationTrainer
from TimeDistill.trainers.chronos2_trainer import Chronos2DistillLossWeights


def main():
    """主函数：使用配置类系统"""
    # 解析命令行参数
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # 处理配置文件路径
    if args.config:
        # 如果指定了配置文件，使用指定路径
        config_path = args.config
        # 如果是相对路径，尝试相对于项目根目录
        if not os.path.isabs(config_path) and not os.path.exists(config_path):
            # 尝试相对于项目根目录
            project_root = Path(__file__).parent.parent.parent
            alt_path = project_root / config_path
            if alt_path.exists():
                config_path = str(alt_path)
    else:
        # 使用默认配置文件
        project_root = Path(__file__).parent.parent.parent
        default_config = project_root / "TimeDistill" / "configs" / "chronos-2-distill.yaml"
        config_path = str(default_config) if default_config.exists() else None
    
    # 加载配置（从YAML或使用默认值，并用命令行参数覆盖）
    config = load_config(config_path=config_path, args=args)
    
    # 设置环境变量
    os.environ.setdefault("HF_ENDPOINT", config.system.hf_endpoint)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.system.cuda_visible_devices)
    
    # 确定设备
    if config.system.device.startswith("cuda") and not torch.cuda.is_available():
        print("警告: 请求使用CUDA但CUDA不可用，切换到CPU")
        device = "cpu"
    else:
        device = config.system.device
    
    if config.system.verbose:
        print("=" * 50)
        print("配置信息")
        print("=" * 50)
        print(f"教师模型: {config.model.teacher_model_id}")
        print(f"数据目录: {config.data.data_dir}")
        print(f"上下文长度: {config.data.context_length}")
        print(f"预测长度: {config.data.horizon}")
        print(f"学习率: {config.training.learning_rate}")
        print(f"批次大小: {config.training.batch_size}")
        print(f"训练轮数: {config.training.num_epochs}")
        print(f"输出目录: {config.system.output_dir}")
        print("=" * 50)
    
    print("=" * 50)
    print("加载教师模型 (Chronos-2)...")
    print("=" * 50)

    teacher_pipeline = Chronos2Pipeline.from_pretrained(
        config.model.teacher_model_id,
        device_map=None,  # 避免 accelerate 自动切分
    )
    try:
        teacher_pipeline.model = teacher_pipeline.model.to("cpu")
    except RuntimeError as e:
        if "accelerate hooks" in str(e).lower():
            print("警告：模型使用了 accelerate hooks，跳过 CPU 移动。将在 trainer 初始化时处理设备。")
        else:
            raise

    teacher_params = sum(p.numel() for p in teacher_pipeline.model.parameters())
    print(f"教师模型加载完成: {config.model.teacher_model_id}")
    print(f"教师模型参数量: {teacher_params:,}")

    print("\n" + "=" * 50)
    print("创建学生模型...")
    print("=" * 50)
    student_model = create_student_model(
        teacher_pipeline, 
        config.model.student_config_overrides
    )
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"学生模型参数量: {student_params:,}")
    print(f"压缩比: {student_params / teacher_params:.2%}")

    print("\n" + "=" * 50)
    print("准备训练/验证数据...")
    print("=" * 50)

    train_dataset = Chronos2DistillationDataset(
        data_dir=config.data.data_dir,
        dataset_names=config.data.dataset_names,
        context_length=config.data.context_length,
        horizon=config.data.horizon,
        stride=config.data.stride,
        split="train",
        train_split=config.data.train_split,
        test_split=config.data.test_split,
        scale=config.data.scale,
        target_col=config.data.target_col,
        time_col_name=config.data.time_col_name,
        verbose=config.system.verbose,
    )
    eval_dataset = Chronos2DistillationDataset(
        data_dir=config.data.data_dir,
        dataset_names=config.data.dataset_names,
        context_length=config.data.context_length,
        horizon=config.data.horizon,
        stride=config.data.horizon,  # 验证集使用horizon作为stride
        split="val",
        train_split=config.data.train_split,
        test_split=config.data.test_split,
        scale=config.data.scale,
        target_col=config.data.target_col,
        time_col_name=config.data.time_col_name,
        verbose=config.system.verbose,
    )

    print(f"\n训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset)}")

    # DataLoader
    pin_memory = config.training.pin_memory if device != "cpu" else False
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers if device != "cpu" else 4,
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers if device != "cpu" else 4,
        pin_memory=pin_memory,
    )

    # 创建损失权重配置
    loss_weights = Chronos2DistillLossWeights(
        alpha_pred_distill=config.loss.alpha_pred_distill,
        beta_feature=config.loss.beta_feature,
        dtw_weight=config.loss.dtw_weight,
    )

    print("\n" + "=" * 50)
    print("启动训练...")
    print("=" * 50)
    print(f"device={device}, output_dir={config.system.output_dir}")
    print(f"loss_weights={asdict(loss_weights)}, dtw_gamma={config.loss.dtw_gamma}")

    trainer = Chronos2DistillationTrainer(
        teacher_pipeline=teacher_pipeline,
        student_model=student_model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        horizon=config.data.horizon,
        learning_rate=config.training.learning_rate,
        num_epochs=config.training.num_epochs,
        device=device,
        output_dir=config.system.output_dir,
        max_grad_norm=config.training.max_grad_norm,
        loss_weights=loss_weights,
        dtw_gamma=config.loss.dtw_gamma,
        use_fast_dtw=config.loss.use_fast_dtw,
        temperature=config.loss.temperature,
        verbose=config.system.verbose,
    )

    # 如需断点续训：
    # trainer.train(resume_checkpoint_path="./chronos-2-distilled/checkpoint_epoch_3")
    trainer.train()

    print("\n" + "=" * 50)
    print("蒸馏完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()


