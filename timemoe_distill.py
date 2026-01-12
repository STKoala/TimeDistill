#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
TimeMoE 知识蒸馏脚本
整合到 TimeDistill 目录，使用 Pretrain_Data 数据训练
支持三层蒸馏：输出层、隐层、路由层
支持 Hook 机制提取注意力前后的特征
"""
import os
import sys
import math
import logging
from pathlib import Path
from datetime import datetime

# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 禁用 wandb（如果不想使用交互式提示）
os.environ["WANDB_DISABLED"] = "true"

# 添加 Time-MoE 路径到 sys.path
timemoe_path = Path(__file__).parent.parent / "Time-MoE"
if timemoe_path.exists():
    sys.path.insert(0, str(timemoe_path))

import torch
import torch.nn as nn
from typing import Optional, List

# 导入 Time-MoE 相关模块
try:
    from time_moe.models.modeling_time_moe import TimeMoeForPrediction, TimeMoeConfig
    from time_moe.models.tiny_time_moe import TinyTimeMoeForPrediction
    from time_moe.datasets.time_moe_dataset import TimeMoEDataset
    from time_moe.datasets.time_moe_window_dataset import TimeMoEWindowDataset
    from time_moe.trainer.distill_trainer import TimeMoEDistillTrainer
    from time_moe.trainer.hf_trainer import TimeMoETrainingArguments
    from time_moe.utils.dist_util import get_world_size
    from time_moe.utils.log_util import logger as timemoe_logger, log_in_local_rank_0
    TIMEMOE_AVAILABLE = True
except ImportError as e:
    TIMEMOE_AVAILABLE = False
    print(f"警告: Time-MoE 模块导入失败: {e}")
    print("请确保 Time-MoE 目录在正确的位置")


def setup_logging(output_dir: str):
    """设置日志记录"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"timemoe_distill_{timestamp}.log"
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    return logger


def create_student_config(teacher_config: TimeMoeConfig, student_config_overrides: dict = None):
    """
    创建学生模型配置（基于教师配置，但更小）
    
    Args:
        teacher_config: 教师模型配置
        student_config_overrides: 学生模型配置覆盖参数
    """
    defaults = {
        'hidden_size': 128,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'intermediate_size': 256,
        'num_experts': 2,
        'num_experts_per_tok': 1,
        'input_size': 1,  # 单变量
    }
    
    if student_config_overrides:
        defaults.update(student_config_overrides)
    
    # 从教师配置继承其他参数
    student_config_dict = teacher_config.to_dict()
    student_config_dict.update(defaults)
    
    # 确保 horizon_lengths 保持一致
    student_config_dict['horizon_lengths'] = teacher_config.horizon_lengths
    
    student_config = TimeMoeConfig.from_dict(student_config_dict)
    return student_config


def find_data_dir():
    """查找数据目录"""
    possible_dirs = [
        "TimeDistill/data/datasets/Pretrain_Data",
        "data/datasets/Pretrain_Data",
        "./data/datasets/Pretrain_Data",
        "../data/datasets/Pretrain_Data",
        str(Path(__file__).parent / "data" / "datasets" / "Pretrain_Data"),
    ]
    
    for data_dir in possible_dirs:
        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            return data_dir
    
    return None


def main():
    """主函数"""
    if not TIMEMOE_AVAILABLE:
        print("错误: Time-MoE 模块未正确导入，请检查路径和依赖")
        return
    
    # ========== 配置参数 ==========
    teacher_model_path = "Maple728/TimeMoE-50M"  # 或本地路径
    output_dir = "./timemoe-distilled"
    
    # 数据配置
    data_dir = find_data_dir()
    if data_dir is None:
        print("错误: 未找到数据目录 Pretrain_Data")
        print("请确保数据目录位于以下位置之一:")
        for d in ["data/datasets/Pretrain_Data", "./data/datasets/Pretrain_Data"]:
            print(f"  - {d}")
        return
    
    print(f"使用数据目录: {data_dir}")
    
    # 训练参数
    max_length = 720  # 输入序列长度
    horizon = 96      # 预测长度
    stride = None     # 滑动窗口步长（None 表示等于 max_length）
    normalization_method = "zero"  # 归一化方法
    
    # 学生模型配置
    student_config_overrides = {
        'hidden_size': 128,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'intermediate_size': 256,
        'num_experts': 2,
        'num_experts_per_tok': 1,
        'input_size': 1,  # 单变量
    }
    
    # 蒸馏参数
    alpha = 1.0   # 输出蒸馏权重
    beta = 0.5    # 隐层蒸馏权重
    gamma = 0.1   # 路由蒸馏权重
    temperature = 1.0  # 温度参数
    lambda_gt = 0.0    # 真实标签损失权重（0=纯蒸馏）
    use_hook = False   # 是否使用 hook
    hook_layers = [0, -1]  # hook 层索引
    
    # 训练配置
    learning_rate = 1e-4
    min_learning_rate = 5e-5
    global_batch_size = 64
    micro_batch_size = 16
    num_train_epochs = 10
    train_steps = None
    precision = "bf16"  # fp32, fp16, bf16
    gradient_checkpointing = False
    weight_decay = 0.1
    warmup_ratio = 0.1
    lr_scheduler_type = "cosine"
    max_grad_norm = 1.0
    logging_steps = 100
    save_steps = 1000
    seed = 9899
    
    # 设置日志
    logger = setup_logging(output_dir)
    logger.info("=" * 80)
    logger.info("TimeMoE 知识蒸馏训练")
    logger.info("=" * 80)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"教师模型: {teacher_model_path}")
    
    # ========== 设置随机种子 ==========
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"随机种子: {seed}")
    
    # ========== 设置精度 ==========
    if precision == 'bf16':
        model_dtype = torch.bfloat16
    elif precision == 'fp16':
        model_dtype = torch.float32
    else:
        model_dtype = torch.float32
    logger.info(f"精度: {precision}")
    
    # ========== 检查模型路径 ==========
    # 检查是否是本地路径
    is_local_path = os.path.exists(teacher_model_path) and os.path.isdir(teacher_model_path)
    if is_local_path:
        logger.info(f"检测到本地模型路径: {teacher_model_path}")
    else:
        logger.info(f"使用 HuggingFace 模型: {teacher_model_path}")
        logger.info("提示: 如果网络连接有问题，可以下载模型到本地，然后使用本地路径")
    
    # ========== 加载教师模型 ==========
    logger.info("=" * 80)
    logger.info("加载教师模型...")
    logger.info("=" * 80)
    try:
        # 构建加载参数
        load_kwargs = {
            "dtype": model_dtype,  # 使用 dtype 而不是 torch_dtype
            "attn_implementation": "eager",
        }
        
        # 如果是本地路径，尝试使用 local_files_only
        if is_local_path:
            load_kwargs["local_files_only"] = True
        
        teacher_model = TimeMoeForPrediction.from_pretrained(
            teacher_model_path,
            **load_kwargs
        )
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        logger.info(f"教师模型加载完成: {teacher_model_path}")
        
        teacher_config = teacher_model.config
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        logger.info(f"教师模型参数量: {teacher_params:,}")
        logger.info(f"教师模型配置: {teacher_config}")
    except Exception as e:
        logger.error(f"加载教师模型失败: {e}")
        logger.error("")
        logger.error("可能的解决方案:")
        logger.error("1. 如果使用 HuggingFace 模型，请检查网络连接")
        logger.error("2. 如果网络受限，可以:")
        logger.error("   - 使用本地模型路径（下载模型后）")
        logger.error("   - 设置环境变量 HF_ENDPOINT 为其他镜像")
        logger.error("   - 使用 local_files_only=True（如果模型已缓存）")
        logger.error("3. 如果使用本地路径，请确保路径正确且包含模型文件")
        logger.error("")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 创建学生模型 ==========
    logger.info("=" * 80)
    logger.info("创建学生模型...")
    logger.info("=" * 80)
    
    student_config = create_student_config(teacher_config, student_config_overrides)
    logger.info(f"学生模型配置: {student_config}")
    
    student_model = TinyTimeMoeForPrediction(student_config)
    student_model = student_model.to(dtype=model_dtype)
    
    student_params = sum(p.numel() for p in student_model.parameters())
    compression_ratio = teacher_params / student_params if student_params > 0 else 0
    
    logger.info(f"学生模型参数量: {student_params:,}")
    logger.info(f"压缩比: {compression_ratio:.2f}x")
    
    # ========== 计算 batch size ==========
    num_devices = get_world_size()
    if micro_batch_size * num_devices > global_batch_size:
        if num_devices > global_batch_size:
            micro_batch_size = 1
            global_batch_size = num_devices
        else:
            micro_batch_size = math.ceil(global_batch_size / num_devices)
    
    gradient_accumulation_steps = math.ceil(global_batch_size / num_devices / micro_batch_size)
    global_batch_size = int(gradient_accumulation_steps * num_devices * micro_batch_size)
    
    logger.info(f"设备数: {num_devices}")
    logger.info(f"Global batch size: {global_batch_size}")
    logger.info(f"Micro batch size: {micro_batch_size}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # ========== 准备数据集 ==========
    logger.info("=" * 80)
    logger.info("准备数据集...")
    logger.info("=" * 80)
    
    try:
        # 创建 CSV 数据集适配器（因为 TimeMoE 原生不支持 CSV）
        try:
            # 添加当前目录到路径以便导入 csv_dataset_adapter
            current_dir = str(Path(__file__).parent)
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            from csv_dataset_adapter import CSVTimeSeriesDataset
            logger.info("使用 CSV 数据集适配器加载 CSV 文件...")
            dataset = CSVTimeSeriesDataset(data_dir, normalization_method=normalization_method)
            logger.info(f"数据集序列数: {len(dataset)}")
        except Exception as adapter_error:
            # 如果适配器不可用，尝试使用原生 TimeMoE 数据集（需要转换数据格式）
            logger.warning(f"CSV 适配器不可用 ({adapter_error})，尝试使用原生 TimeMoE 数据集（可能不支持 CSV）...")
            dataset = TimeMoEDataset(data_dir, normalization_method=normalization_method)
            logger.info(f"数据集序列数: {len(dataset)}")
            if len(dataset) == 0:
                logger.error("数据集为空！TimeMoE 原生数据集不支持 CSV 格式，请确保 csv_dataset_adapter.py 可用")
                raise ValueError("数据集为空，无法继续训练")
        
        # 创建滑动窗口数据集
        stride = stride if stride is not None else max_length
        train_dataset = TimeMoEWindowDataset(
            dataset,
            context_length=max_length,
            prediction_length=0,
            stride=stride,
            shuffle=False
        )
        logger.info(f"训练样本数: {len(train_dataset)}")
        logger.info(f"Context length: {max_length}")
        logger.info(f"Horizon: {horizon}")
        logger.info(f"Stride: {stride}")
    except Exception as e:
        logger.error(f"数据集准备失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 训练参数 ==========
    train_steps_val = train_steps if train_steps is not None and train_steps > 0 else -1
    num_train_epochs_val = num_train_epochs if train_steps_val < 0 else -1
    
    training_args = TimeMoETrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs_val,
        max_steps=train_steps_val,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        bf16=(precision == 'bf16'),
        fp16=(precision == 'fp16'),
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_strategy='steps',
        logging_dir=os.path.join(output_dir, 'tb_logs'),
        seed=seed,
        data_seed=seed,
        max_grad_norm=max_grad_norm,
        dataloader_num_workers=4,
        save_only_model=True,
        report_to=[],  # 禁用 wandb 等外部日志记录工具，只使用 tensorboard（通过 logging_dir）
    )
    
    # ========== 处理 hook_layers ==========
    hook_layers_processed = hook_layers if use_hook else []
    if hook_layers_processed:
        num_layers = student_config.num_hidden_layers
        hook_layers_processed = [i if i >= 0 else num_layers + i for i in hook_layers_processed]
        hook_layers_processed = [i for i in hook_layers_processed if 0 <= i < num_layers]
    
    logger.info("=" * 80)
    logger.info("蒸馏配置")
    logger.info("=" * 80)
    logger.info(f"输出蒸馏权重 (alpha): {alpha}")
    logger.info(f"隐层蒸馏权重 (beta): {beta}")
    logger.info(f"路由蒸馏权重 (gamma): {gamma}")
    logger.info(f"温度参数: {temperature}")
    logger.info(f"真实标签权重 (lambda_gt): {lambda_gt}")
    logger.info(f"使用 Hook: {use_hook}")
    if use_hook:
        logger.info(f"Hook layers: {hook_layers_processed}")
    
    # ========== 创建蒸馏训练器 ==========
    logger.info("=" * 80)
    logger.info("创建蒸馏训练器...")
    logger.info("=" * 80)
    
    try:
        trainer = TimeMoEDistillTrainer(
            teacher_model=teacher_model,
            student_model=student_model,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            temperature=temperature,
            lambda_gt=lambda_gt,
            use_hook=use_hook,
            hook_layers=hook_layers_processed,
            args=training_args,
            train_dataset=train_dataset,
        )
        logger.info("蒸馏训练器创建成功")
    except Exception as e:
        logger.error(f"创建蒸馏训练器失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 开始训练 ==========
    logger.info("=" * 80)
    logger.info("开始蒸馏训练...")
    logger.info("=" * 80)
    
    try:
        trainer.train()
        logger.info("训练完成！")
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 保存模型 ==========
    logger.info("=" * 80)
    logger.info("保存模型...")
    logger.info("=" * 80)
    try:
        trainer.save_model(output_dir)
        logger.info(f"模型已保存到: {output_dir}")
    except Exception as e:
        logger.error(f"保存模型失败: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("=" * 80)
    logger.info("所有任务完成！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

