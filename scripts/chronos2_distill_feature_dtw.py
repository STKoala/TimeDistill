#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chronos-2 蒸馏入口脚本（特征蒸馏 + DTW）

用法（示例）：
  python -m TimeDistill.scripts.chronos2_distill_feature_dtw

说明：
- 该脚本仅负责“组装配置/创建数据集与DataLoader/创建trainer并启动训练”
- 训练主逻辑在 `TimeDistill.trainers.Chronos2DistillationTrainer`
"""

from __future__ import annotations

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dataclasses import asdict
from typing import Optional

import torch
from chronos import Chronos2Pipeline
from torch.utils.data import DataLoader

from datasets import Chronos2DistillationDataset
from models import create_student_model
from trainers import Chronos2DistillationTrainer
from trainers.chronos2_trainer import Chronos2DistillLossWeights


def main(
    teacher_model_id: str = "amazon/chronos-2",
    data_dir: str = "data/datasets/Pretrain_Data",
    dataset_names: Optional[list[str]] = None,
    output_dir: str = "./chronos-2-distilled",
):
    # 环境：镜像与单卡
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 数据配置（参考你原脚本默认）
    train_split = 0.8
    test_split = 0.0
    context_length = 720
    horizon = 96
    stride = 96

    # 学生模型配置
    student_config_overrides = {
        "num_layers": 6,
        "d_model": 384,
        "d_ff": 1536,
        "num_heads": 6,
    }

    # 训练配置
    learning_rate = 1e-5
    batch_size = 256
    num_epochs = 20
    max_grad_norm = 1.0

    # loss 权重
    loss_weights = Chronos2DistillLossWeights(
        alpha_pred_distill=0.5,
        beta_feature=0.3,
        dtw_weight=0.2,
    )
    dtw_gamma = 1.0

    print("=" * 50)
    print("加载教师模型 (Chronos-2)...")
    print("=" * 50)

    teacher_pipeline = Chronos2Pipeline.from_pretrained(
        teacher_model_id,
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
    print(f"教师模型加载完成: {teacher_model_id}")
    print(f"教师模型参数量: {teacher_params:,}")

    print("\n" + "=" * 50)
    print("创建学生模型...")
    print("=" * 50)
    student_model = create_student_model(teacher_pipeline, student_config_overrides)
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"学生模型参数量: {student_params:,}")
    print(f"压缩比: {student_params / teacher_params:.2%}")

    print("\n" + "=" * 50)
    print("准备训练/验证数据...")
    print("=" * 50)

    train_dataset = Chronos2DistillationDataset(
        data_dir=data_dir,
        dataset_names=dataset_names,
        context_length=context_length,
        horizon=horizon,
        stride=stride,
        split="train",
        train_split=train_split,
        test_split=test_split,
        scale=True,
    )
    eval_dataset = Chronos2DistillationDataset(
        data_dir=data_dir,
        dataset_names=dataset_names,
        context_length=context_length,
        horizon=horizon,
        stride=horizon,
        split="val",
        train_split=train_split,
        test_split=test_split,
        scale=True,
    )

    print(f"\n训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset)}")

    # DataLoader（GPU 时 num_workers=0 更稳）
    pin_memory = device == "cpu"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if device != "cpu" else 4,
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if device != "cpu" else 4,
        pin_memory=pin_memory,
    )

    print("\n" + "=" * 50)
    print("启动训练...")
    print("=" * 50)
    print(f"device={device}, output_dir={output_dir}")
    print(f"loss_weights={asdict(loss_weights)}, dtw_gamma={dtw_gamma}")

    trainer = Chronos2DistillationTrainer(
        teacher_pipeline=teacher_pipeline,
        student_model=student_model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        horizon=horizon,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        device=device,
        output_dir=output_dir,
        max_grad_norm=max_grad_norm,
        loss_weights=loss_weights,
        dtw_gamma=dtw_gamma,
        use_fast_dtw=True,
        verbose=True,
    )

    # 如需断点续训：
    # trainer.train(resume_checkpoint_path="./chronos-2-distilled/checkpoint_epoch_3")
    trainer.train()

    print("\n" + "=" * 50)
    print("蒸馏完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()


