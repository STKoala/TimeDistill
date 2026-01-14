"""
Chronos-2 蒸馏简化示例
这是一个最小化的示例，展示如何使用手动蒸馏方法训练 Chronos-2 学生模型
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from chronos import Chronos2Pipeline, Chronos2Model
from TimeDistill.datasets import Chronos2DistillationDataset
from TimeDistill.models import create_student_model
from TimeDistill.trainers import Chronos2DistillationTrainer
from torch.utils.data import DataLoader

def main():
    """简化的蒸馏示例"""
    
    # 1. 加载教师模型
    print("加载教师模型...")
    teacher_pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-bolt-base",
        device_map="auto"
    )
    
    # 2. 创建学生模型（更小的架构）
    print("创建学生模型...")
    student_model = create_student_model(
        teacher_pipeline,
        student_config_overrides={
            "num_layers": 3,    # 减少层数
            "d_model": 384,     # 减少隐藏维度
            "d_ff": 1536,       # 减少前馈维度
            "num_heads": 6,     # 减少注意力头
        }
    )
    
    # 3. 准备数据
    print("准备数据...")
    train_dataset = Chronos2DistillationDataset(
        data_paths=["datasets/ETTh1.csv"],  # 使用单个数据集作为示例
        context_length=96,
        horizon=24,
        stride=1
    )
    
    # 4. 创建训练器并训练
    print("开始训练...")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    trainer = Chronos2DistillationTrainer(
        teacher_pipeline=teacher_pipeline,
        student_model=student_model,
        train_loader=train_loader,
        eval_loader=None,
        horizon=24,
        learning_rate=5e-5,
        num_epochs=5,
        temperature=2.0,
        output_dir="./chronos-2-distilled-simple",
    )
    
    trainer.train()
    print("训练完成！")

if __name__ == "__main__":
    main()

