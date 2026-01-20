"""
TimesFM 从头开始训练脚本

直接使用 TimesFM 官方模型类进行训练：
- 使用官方的 TimesFM_2p5_200M_torch_module 模型
- 使用官方的数据处理函数（strip_leading_nans, linear_interpolation）
- 符合官方数据格式（patch-based inputs + masks）
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
# 添加 timesfm 源码路径
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timesfm_src_path = os.path.join(script_dir, '..', 'timesfm', 'src')
    if os.path.exists(timesfm_src_path):
        sys.path.insert(0, timesfm_src_path)
except:
    timesfm_src_path = os.path.join('..', 'timesfm', 'src')
    if os.path.exists(timesfm_src_path):
        sys.path.insert(0, timesfm_src_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import argparse

# 导入 TimesFM 官方组件
try:
    from timesfm.timesfm_2p5.timesfm_2p5_base import strip_leading_nans, linear_interpolation
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
    from timesfm.torch import util
    print("成功导入 TimesFM 官方组件")
except ImportError:
    try:
        # 尝试从已安装的包导入
        from timesfm.timesfm_2p5.timesfm_2p5_base import strip_leading_nans, linear_interpolation
        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
        from timesfm.torch import util
        print("成功导入 TimesFM 官方组件（从已安装包）")
    except ImportError as e:
        print(f"错误: 无法导入 TimesFM 官方组件: {e}")
        print("请确保已安装 timesfm 包: pip install timesfm")
        raise

warnings.filterwarnings('ignore')


class TimesFMTrainingDataset(Dataset):
    """TimesFM 训练数据集（使用官方数据处理）"""
    
    def __init__(
        self,
        data_paths: List[str] = None,
        context_length: int = 96,
        horizon: int = 24,
        stride: int = 1,
        target_col: str = None,
        data_dir: str = None,
        dataset_names: List[str] = None,
        scale: bool = True,
        train_split: float = 0.8,
        test_split: float = 0.1,
        split: str = 'train',
        time_col_name: str = 'date',
        max_files: int = 100,
        max_samples_per_file: int = 1000,
        patch_len: int = 32  # TimesFM 官方 patch 长度
    ):
        self.context_length = context_length
        self.horizon = horizon
        self.stride = stride
        self.scale = scale
        self.split = split
        self.time_col_name = time_col_name
        self.max_files = max_files
        self.max_samples_per_file = max_samples_per_file
        self.train_split = train_split
        self.test_split = test_split
        self.patch_len = patch_len
        
        # 存储原始数据和元数据
        self.data_segments = []
        self.sample_offsets = []
        self.scalers = {}
        
        # 如果提供了数据集名称列表，从目录中查找对应的 CSV 文件
        if dataset_names and data_dir:
            print(f"从目录加载指定数据集: {data_dir}")
            data_paths = []
            for name in dataset_names:
                possible_names = [
                    name + ".csv",
                    name.lower() + ".csv",
                    name.replace("_", "-") + ".csv",
                ]
                found = False
                for possible_name in possible_names:
                    full_path = os.path.join(data_dir, possible_name)
                    if os.path.exists(full_path):
                        data_paths.append(full_path)
                        found = True
                        break
                if not found:
                    print(f"  警告: 未找到数据集 {name}，跳过")
        
        # 如果提供了数据目录但没有指定数据集名称，加载所有 CSV 文件
        elif data_dir and os.path.isdir(data_dir) and not dataset_names:
            print(f"从目录加载所有 CSV 文件: {data_dir}")
            csv_files = list(Path(data_dir).glob("*.csv"))
            print(f"找到 {len(csv_files)} 个 CSV 文件")
            
            if len(csv_files) > self.max_files:
                print(f"警告: 文件数量过多 ({len(csv_files)})，将只处理前 {self.max_files} 个文件")
                csv_files = csv_files[:self.max_files]
            
            data_paths = [str(f) for f in csv_files]
        
        if not data_paths:
            data_paths = data_paths or []
        
        # 展开 data_paths 中的目录为文件列表
        expanded_data_paths = []
        for path in data_paths:
            if os.path.isdir(path):
                csv_files = list(Path(path).glob("*.csv"))
                if csv_files:
                    print(f"从目录 {path} 找到 {len(csv_files)} 个 CSV 文件")
                    expanded_data_paths.extend([str(f) for f in csv_files])
                else:
                    print(f"警告: 目录 {path} 中没有找到 CSV 文件，跳过")
            elif os.path.isfile(path):
                expanded_data_paths.append(path)
            else:
                print(f"警告: 路径 {path} 不存在，跳过")
        
        data_paths = expanded_data_paths
        
        # 处理数据文件
        total_files = len(data_paths)
        print(f"开始处理 {total_files} 个文件...")
        for file_idx, data_path in enumerate(data_paths):
            if (file_idx + 1) % 10 == 0 or (file_idx + 1) == total_files:
                print(f"  处理进度: {file_idx + 1}/{total_files} ({100*(file_idx+1)/total_files:.1f}%)")
            if not os.path.exists(data_path):
                print(f"警告: 未找到数据文件 {data_path}，跳过")
                continue
            
            try:
                df = pd.read_csv(data_path)
                
                # 确定目标列
                exclude_cols = ['date', 'timestamp', 'time', 'id', 'item_id', 'time_idx']
                numeric_cols = [col for col in df.columns 
                               if col.lower() not in [c.lower() for c in exclude_cols]
                               and pd.api.types.is_numeric_dtype(df[col])]
                
                if len(numeric_cols) == 0:
                    target_col_actual = df.columns[-1]
                elif target_col and target_col in df.columns:
                    target_col_actual = target_col
                else:
                    target_col_actual = numeric_cols[0]
                
                # 提取数据（排除时间列）
                if self.time_col_name in df.columns:
                    df_data = df.drop(columns=[self.time_col_name])
                else:
                    df_data = df
                
                df_data = df_data.select_dtypes(include=[np.number])
                
                if len(df_data.columns) == 0:
                    print(f"  跳过 {os.path.basename(data_path)}: 没有数值列")
                    continue
                
                # 使用第一个数值列作为目标（单变量预测）
                ts_values_raw = df_data.iloc[:, 0].values.astype(np.float32)
                
                # 使用官方数据处理函数预处理
                ts_values_raw = strip_leading_nans(ts_values_raw)
                ts_values_raw = linear_interpolation(ts_values_raw)
                
                if len(ts_values_raw) == 0:
                    print(f"  跳过 {os.path.basename(data_path)}: 没有有效数据")
                    continue
                
                if np.isinf(ts_values_raw).any():
                    print(f"  跳过 {os.path.basename(data_path)}: 包含 Inf 值")
                    continue
                
                # 转换为 2D 数组以兼容 StandardScaler
                data = ts_values_raw.reshape(-1, 1)
                
                # 数据划分
                num_train = int(len(data) * train_split)
                num_test = int(len(data) * test_split)
                num_val = len(data) - num_train - num_test
                
                border1s = [0, num_train - context_length, len(data) - num_test - context_length]
                border2s = [num_train, num_train + num_val, len(data)]
                
                split_map = {'train': 0, 'val': 1, 'test': 2}
                split_idx = split_map.get(split, 0)
                border1 = border1s[split_idx]
                border2 = border2s[split_idx]
                
                split_data = data[border1:border2]
                
                if len(split_data) == 0:
                    print(f"  跳过 {os.path.basename(data_path)}: {split} 集为空")
                    continue
                
                # 数据标准化
                if self.scale:
                    train_data = data[border1s[0]:border2s[0]]
                    scaler = StandardScaler()
                    scaler.fit(train_data)
                    split_data = scaler.transform(split_data)
                    self.scalers[os.path.basename(data_path)] = scaler
                
                # 提取单变量时间序列
                ts_values = split_data[:, 0].astype(np.float32)
                
                # 再次使用官方数据处理函数确保数据质量
                ts_values = strip_leading_nans(ts_values)
                ts_values = linear_interpolation(ts_values)
                
                # TimesFM 需要 context_length 是 patch_len 的倍数
                # 调整 context_length 使其对齐到 patch 边界
                adjusted_context = (context_length // self.patch_len) * self.patch_len
                if adjusted_context < self.patch_len:
                    adjusted_context = self.patch_len
                
                if len(ts_values) < adjusted_context + horizon:
                    print(f"  跳过 {os.path.basename(data_path)}: {split} 集数据长度不足")
                    continue
                
                max_start_idx = len(ts_values) - adjusted_context - horizon
                
                num_samples = (max_start_idx // stride) + 1
                if num_samples > self.max_samples_per_file:
                    num_samples = self.max_samples_per_file
                
                segment_idx = len(self.data_segments)
                scaler = self.scalers.get(os.path.basename(data_path), None) if self.scale else None
                self.data_segments.append({
                    'values': ts_values,
                    'scaler': scaler,
                    'file_name': os.path.basename(data_path),
                    'stride': stride,
                    'max_start_idx': max_start_idx,
                    'adjusted_context': adjusted_context
                })
                
                sample_count = 0
                for start_idx in range(0, max_start_idx + 1, stride):
                    if sample_count >= num_samples:
                        break
                    self.sample_offsets.append((segment_idx, start_idx))
                    sample_count += 1
                
                if sample_count > 0:
                    print(f"  {os.path.basename(data_path)} ({split}): 可生成 {sample_count} 个样本")
                    
            except Exception as e:
                print(f"  处理 {os.path.basename(data_path)} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n共准备 {len(self.sample_offsets)} 个 {split} 样本")
    
    def __len__(self):
        return len(self.sample_offsets)
    
    def __getitem__(self, idx):
        if idx >= len(self.sample_offsets):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.sample_offsets)})")
        
        segment_idx, start_idx = self.sample_offsets[idx]
        segment = self.data_segments[segment_idx]
        ts_values = segment['values']
        adjusted_context = segment['adjusted_context']
        
        context = ts_values[start_idx:start_idx + adjusted_context]
        target = ts_values[start_idx + adjusted_context:start_idx + adjusted_context + self.horizon]
        
        # 使用官方数据处理函数处理 NaN
        context = linear_interpolation(context)
        target = linear_interpolation(target)
        
        # 创建 mask（TimesFM 官方格式：False 表示有效值，True 表示缺失值）
        context_mask = np.isnan(context) | np.isinf(context)
        
        # 填充 NaN/Inf
        context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        
        return {
            "inputs": torch.tensor(context, dtype=torch.float32),
            "masks": torch.tensor(context_mask, dtype=torch.bool),
            "targets": torch.tensor(target, dtype=torch.float32)
        }


class TimesFMTrainer:
    """TimesFM 训练器（使用官方模型）"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: TimesFMTrainingDataset,
        eval_dataset: Optional[TimesFMTrainingDataset] = None,
        context_length: int = 96,
        horizon: int = 24,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        num_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./timesfm-trained",
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 0,
        save_steps: int = 1000,
        eval_steps: int = 500,
        logging_steps: int = 100,
        lr_scheduler_type: Optional[str] = "reduce_on_plateau",
        lr_scheduler_patience: int = 2,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_min_lr: float = 1e-6
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.context_length = context_length
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.output_dir = output_dir
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.lr_scheduler_type = lr_scheduler_type
        
        # 将模型移到设备
        self.model = self.model.to(device)
        self.model.train()  # 设置为训练模式
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            eps=1e-8
        )
        
        # 学习率调度器
        if lr_scheduler_type is None:
            self.scheduler = None
            self.scheduler_step_on_epoch = False
        elif lr_scheduler_type == "reduce_on_plateau":
            # 基于验证损失的学习率衰减
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=lr_scheduler_factor,
                patience=lr_scheduler_patience,
                min_lr=lr_scheduler_min_lr
            )
            self.scheduler_step_on_epoch = True  # 在epoch结束时调用
        elif lr_scheduler_type == "cosine":
            # 余弦退火调度器
            from torch.optim.lr_scheduler import CosineAnnealingLR
            total_steps = num_epochs * len(train_dataset) // (batch_size * gradient_accumulation_steps)
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=lr_scheduler_min_lr
            )
            self.scheduler_step_on_epoch = False  # 在step时调用
        elif warmup_steps > 0:
            # Warmup调度器
            from torch.optim.lr_scheduler import LambdaLR
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
            self.scheduler_step_on_epoch = False
        else:
            self.scheduler = None
            self.scheduler_step_on_epoch = False
        
        # 数据加载器
        pin_memory = device != "cpu"
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0 if device != "cpu" else 4,
            pin_memory=pin_memory
        )
        
        if eval_dataset:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0 if device != "cpu" else 4,
                pin_memory=pin_memory
            )
        else:
            self.eval_loader = None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 训练状态
        self.global_step = 0
        self.best_eval_loss = float('inf')
    
    def train_epoch(self, epoch: int):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs = batch["inputs"].to(self.device)  # [batch_size, context_length]
            masks = batch["masks"].to(self.device)  # [batch_size, context_length]
            targets = batch["targets"].to(self.device)  # [batch_size, horizon]
            
            # 只在梯度累积开始时清零梯度
            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            try:
                # TimesFM 官方模型期望输入是 patch 格式
                # 需要将 [batch_size, context_length] reshape 成 [batch_size, num_patches, patch_len]
                batch_size, context_len = inputs.shape[0], inputs.shape[1]
                patch_len = self.model.p  # 32
                
                # 确保 context_length 是 patch_len 的倍数
                if context_len % patch_len != 0:
                    # 截断到最近的 patch_len 倍数
                    context_len = (context_len // patch_len) * patch_len
                    inputs = inputs[:, :context_len]
                    masks = masks[:, :context_len]
                
                num_patches = context_len // patch_len
                
                # Reshape 成 patch 格式: [batch_size, num_patches, patch_len]
                patched_inputs = inputs.reshape(batch_size, num_patches, patch_len)
                patched_masks = masks.reshape(batch_size, num_patches, patch_len)
                
                # 官方模型 forward：返回 (input_embeddings, output_embeddings, output_ts, output_quantile_spread), decode_caches
                # output_ts 形状: [batch_size, num_patches, 1280] = [batch_size, num_patches, output_patch_len * quantiles]
                # 需要 reshape 成 [batch_size, num_patches, output_patch_len, quantiles]
                (_, _, output_ts, output_quantile_spread), _ = self.model(patched_inputs, patched_masks, None)
                
                # #region agent log
                with open('/root/shengyuan/Distillation/.cursor/debug.log', 'a') as f:
                    import json
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A",
                        "location": "train_timesfm_from_scratch.py:442",
                        "message": "output_ts shape check before reshape",
                        "data": {
                            "output_ts_shape": list(output_ts.shape),
                            "output_ts_ndim": output_ts.ndim,
                            "batch_size": batch_size,
                            "num_patches": num_patches,
                            "model_o": int(self.model.o),
                            "model_q": int(self.model.q)
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }
                    f.write(json.dumps(log_entry) + '\n')
                # #endregion
                
                # Reshape output_ts: [batch_size, num_patches, 1280] -> [batch_size, num_patches, output_patch_len, quantiles]
                output_patch_len = self.model.o  # 128
                quantiles = self.model.q  # 10
                
                # #region agent log
                with open('/root/shengyuan/Distillation/.cursor/debug.log', 'a') as f:
                    import json
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "B",
                        "location": "train_timesfm_from_scratch.py:470",
                        "message": "before reshape check",
                        "data": {
                            "output_ts_shape": list(output_ts.shape),
                            "output_ts_size": int(output_ts.numel()),
                            "expected_reshape": [batch_size, num_patches, output_patch_len, quantiles],
                            "expected_size": batch_size * num_patches * output_patch_len * quantiles,
                            "output_patch_len": output_patch_len,
                            "quantiles": quantiles
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }
                    f.write(json.dumps(log_entry) + '\n')
                # #endregion
                
                # 验证 reshape 是否可行
                expected_size = batch_size * num_patches * output_patch_len * quantiles
                actual_size = output_ts.numel()
                if expected_size != actual_size:
                    raise ValueError(f"无法 reshape: 期望大小 {expected_size}, 实际大小 {actual_size}, "
                                   f"output_ts.shape={output_ts.shape}, "
                                   f"期望形状=[{batch_size}, {num_patches}, {output_patch_len}, {quantiles}]")
                
                output_ts = output_ts.reshape(batch_size, num_patches, output_patch_len, quantiles)
                
                # #region agent log
                with open('/root/shengyuan/Distillation/.cursor/debug.log', 'a') as f:
                    import json
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A",
                        "location": "train_timesfm_from_scratch.py:460",
                        "message": "output_ts shape check after reshape",
                        "data": {
                            "output_ts_shape": list(output_ts.shape),
                            "output_ts_ndim": output_ts.ndim
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }
                    f.write(json.dumps(log_entry) + '\n')
                # #endregion
                
                # #region agent log
                with open('/root/shengyuan/Distillation/.cursor/debug.log', 'a') as f:
                    import json
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "C",
                        "location": "train_timesfm_from_scratch.py:522",
                        "message": "before accessing output_ts",
                        "data": {
                            "output_ts_shape": list(output_ts.shape),
                            "output_ts_ndim": output_ts.ndim,
                            "will_access": "output_ts[:, -1, :, 5]"
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }
                    f.write(json.dumps(log_entry) + '\n')
                # #endregion
                
                # 提取最后一个 patch 的中间分位数（通常是索引 5，即中位数）
                if output_ts.ndim != 4:
                    raise ValueError(f"output_ts 维度错误: 期望 4 维，实际 {output_ts.ndim} 维，形状={output_ts.shape}")
                
                last_patch_pred = output_ts[:, -1, :, 5]  # 使用中位数分位数 [batch_size, output_patch_len]
                
                # 截取到 horizon 长度
                predictions = last_patch_pred[:, :self.horizon]  # [batch_size, horizon]
                
            except Exception as e:
                print(f"警告: 前向传播失败: {e}")
                import traceback
                traceback.print_exc()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                continue
            
            # 确保形状匹配
            if predictions.shape != targets.shape:
                min_len = min(predictions.shape[1], targets.shape[1])
                predictions = predictions[:, :min_len]
                targets = targets[:, :min_len]
            
            # 检查是否有 NaN 或 Inf 值
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                print(f"警告: Batch {batch_idx + 1} 模型输出包含 NaN/Inf，跳过")
                continue
            
            # 计算损失（MSE）
            loss = F.mse_loss(predictions, targets)
            
            # 检查损失是否为 NaN 或 Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: Batch {batch_idx + 1} 损失包含 NaN/Inf，跳过")
                continue
            
            # 保存原始损失值
            original_loss = loss.item()
            
            # 限制损失值
            if original_loss > 1e6:
                print(f"警告: Batch {batch_idx + 1} 损失过大 ({original_loss:.2f})，跳过")
                continue
            
            # 反向传播（除以累积步数以平均梯度）
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            # 只在累积步数达到时更新参数
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 检查梯度
                has_nan_grad = False
                for param in self.model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break
                
                if has_nan_grad:
                    print(f"警告: Batch {batch_idx + 1} 梯度包含 NaN/Inf，跳过更新")
                    self.optimizer.zero_grad()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    continue
                
                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # 更新参数
                self.optimizer.step()
                
                # 更新学习率（只在非ReduceLROnPlateau时在step时调用）
                if self.scheduler is not None and not self.scheduler_step_on_epoch:
                    self.scheduler.step()
                
                self.global_step += 1
                
                # 清理 GPU 缓存
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # 定期评估
                if self.eval_loader and self.global_step % self.eval_steps == 0:
                    eval_loss = self.evaluate()
                    if eval_loss is not None:
                        print(f"  [Step {self.global_step}] 验证损失: {eval_loss:.4f}")
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_model(f"checkpoint-step-{self.global_step}")
                            print(f"  保存最佳模型 (验证损失: {eval_loss:.4f})")
                
                # 定期保存
                if self.global_step % self.save_steps == 0:
                    self.save_model(f"checkpoint-step-{self.global_step}")
                
                # 定期打印日志
                if self.global_step % self.logging_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"  [Step {self.global_step}] Loss: {original_loss:.4f}, "
                          f"Grad Norm: {grad_norm.item():.4f}, LR: {current_lr:.2e}")
            
            total_loss += original_loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def evaluate(self):
        """评估模型"""
        if self.eval_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                inputs = batch["inputs"].to(self.device)
                masks = batch["masks"].to(self.device)
                targets = batch["targets"].to(self.device)
                
                try:
                    # TimesFM 官方模型期望输入是 patch 格式
                    batch_size, context_len = inputs.shape[0], inputs.shape[1]
                    patch_len = self.model.p  # 32
                    
                    # 确保 context_length 是 patch_len 的倍数
                    if context_len % patch_len != 0:
                        context_len = (context_len // patch_len) * patch_len
                        inputs = inputs[:, :context_len]
                        masks = masks[:, :context_len]
                    
                    num_patches = context_len // patch_len
                    
                    # Reshape 成 patch 格式
                    patched_inputs = inputs.reshape(batch_size, num_patches, patch_len)
                    patched_masks = masks.reshape(batch_size, num_patches, patch_len)
                    
                    (_, _, output_ts, _), _ = self.model(patched_inputs, patched_masks, None)
                    
                    # Reshape output_ts: [batch_size, num_patches, 1280] -> [batch_size, num_patches, output_patch_len, quantiles]
                    output_patch_len = self.model.o  # 128
                    quantiles = self.model.q  # 10
                    output_ts = output_ts.reshape(batch_size, num_patches, output_patch_len, quantiles)
                    
                    last_patch_pred = output_ts[:, -1, :, 5]  # 使用中位数分位数
                    predictions = last_patch_pred[:, :self.horizon]
                    
                    min_len = min(predictions.shape[1], targets.shape[1])
                    loss = F.mse_loss(predictions[:, :min_len], targets[:, :min_len])
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    print(f"评估时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train(self):
        """执行训练"""
        print("=" * 50)
        print("开始 TimesFM 从头训练（使用官方模型）")
        print("=" * 50)
        print(f"训练样本数: {len(self.train_dataset)}")
        if self.eval_dataset:
            print(f"验证样本数: {len(self.eval_dataset)}")
        print(f"总批次数: {len(self.train_loader)}")
        print(f"设备: {self.device}")
        print("=" * 50)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)
            
            train_loss = self.train_epoch(epoch)
            print(f"训练损失: {train_loss:.4f}")
            
            eval_loss = None
            if self.eval_loader:
                eval_loss = self.evaluate()
                if eval_loss is not None:
                    print(f"验证损失: {eval_loss:.4f}")
                    
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_model(f"best_model_epoch_{epoch + 1}")
                        print(f"保存最佳模型 (验证损失: {eval_loss:.4f})")
            
            # 更新学习率调度器（ReduceLROnPlateau需要在epoch结束时调用）
            if self.scheduler is not None and self.scheduler_step_on_epoch:
                if eval_loss is not None:
                    self.scheduler.step(eval_loss)
                else:
                    self.scheduler.step(train_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"当前学习率: {current_lr:.2e}")
            
            # 每个 epoch 结束时保存
            self.save_model(f"checkpoint-epoch-{epoch + 1}")
        
        self.save_model("final_model")
        print("\n训练完成！")
    
    def save_model(self, model_name: str):
        """保存模型"""
        save_path = os.path.join(self.output_dir, model_name)
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型状态
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'context_length': self.context_length,
            'horizon': self.horizon,
            'global_step': self.global_step,
            'best_eval_loss': self.best_eval_loss,
        }, os.path.join(save_path, 'pytorch_model.bin'))
        
        # 保存模型配置
        config = {
            'context_length': self.context_length,
            'horizon': self.horizon,
            'model_type': 'TimesFM_2p5_200M_torch',
            'patch_len': self.model.p,
            'output_patch_len': self.model.o,
            'num_layers': self.model.x,
            'num_heads': self.model.h,
            'model_dims': self.model.md,
        }
        
        import json
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"模型已保存到: {save_path}")


def create_model(
    context_length: int = 96,
    horizon: int = 24,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> TimesFM_2p5_200M_torch_module:
    """
    创建 TimesFM 官方模型
    
    Parameters
    ----------
    context_length : int
        上下文长度（会被调整为 patch_len 的倍数）
    horizon : int
        预测长度
    device : str
        设备
        
    Returns
    -------
    TimesFM_2p5_200M_torch_module
        官方模型实例
    """
    model = TimesFM_2p5_200M_torch_module()
    model = model.to(device)
    return model


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TimesFM 从头开始训练（使用官方模型）")
    
    # 数据配置
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据目录路径")
    parser.add_argument("--context_length", type=int, default=720,
                        help="上下文长度（会被调整为 patch_len=32 的倍数）")
    parser.add_argument("--horizon", type=int, default=96,
                        help="预测长度")
    parser.add_argument("--stride", type=int, default=2,
                        help="滑动窗口步长")
    parser.add_argument("--max_files", type=int, default=50,
                        help="最大处理文件数")
    parser.add_argument("--max_samples_per_file", type=int, default=1000,
                        help="每个文件的最大样本数")
    
    # 训练配置
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪阈值")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="预热步数")
    parser.add_argument("--lr_scheduler_type", type=str, default="reduce_on_plateau",
                        choices=["reduce_on_plateau", "cosine", "none"],
                        help="学习率调度器类型: reduce_on_plateau(基于验证损失), cosine(余弦退火), none(无)")
    parser.add_argument("--lr_scheduler_patience", type=int, default=2,
                        help="ReduceLROnPlateau的patience（验证损失不下降的epoch数）")
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5,
                        help="学习率衰减因子")
    parser.add_argument("--lr_scheduler_min_lr", type=float, default=1e-6,
                        help="最小学习率")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="保存步数")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="评估步数")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="日志打印步数")
    
    # 其他配置
    parser.add_argument("--output_dir", type=str, default="./timesfm-trained",
                        help="输出目录")
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # 查找数据目录
    if args.data_dir is None:
        possible_data_dirs = [
            "data/datasets/Pretrain_Data",
            "datasets/monash_csv_downsmp",
            "data/datasets/monash_csv_downsmp",
            "./datasets/monash_csv_downsmp",
        ]
        train_data_dir = None
        for data_dir in possible_data_dirs:
            if os.path.exists(data_dir) and os.path.isdir(data_dir):
                train_data_dir = data_dir
                print(f"找到数据目录: {train_data_dir}")
                break
        
        if train_data_dir is None:
            print("错误: 未找到数据目录。请使用 --data_dir 指定数据目录。")
            return
    else:
        train_data_dir = args.data_dir
        if not os.path.exists(train_data_dir):
            print(f"错误: 数据目录不存在: {train_data_dir}")
            return
    
    # ========== 创建官方模型 ==========
    print("\n" + "=" * 50)
    print("创建 TimesFM 官方模型...")
    print("=" * 50)
    
    model = create_model(
        context_length=args.context_length,
        horizon=args.horizon,
        device=device
    )
    
    model_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {model_params:,}")
    print(f"Patch 长度: {model.p}")
    print(f"输出 Patch 长度: {model.o}")
    print(f"Transformer 层数: {model.x}")
    
    # 调整 context_length 使其是 patch_len 的倍数
    adjusted_context = (args.context_length // model.p) * model.p
    if adjusted_context < model.p:
        adjusted_context = model.p
    if adjusted_context != args.context_length:
        print(f"注意: context_length 已从 {args.context_length} 调整为 {adjusted_context} (patch_len={model.p} 的倍数)")
    
    # ========== 准备数据 ==========
    print("\n" + "=" * 50)
    print("准备训练数据...")
    print("=" * 50)
    
    train_dataset = TimesFMTrainingDataset(
        data_paths=[],
        data_dir=train_data_dir,
        dataset_names=None,
        context_length=adjusted_context,
        horizon=args.horizon,
        stride=args.stride,
        target_col=None,
        scale=True,
        split='train',
        train_split=0.8,
        test_split=0.1,
        max_files=args.max_files,
        max_samples_per_file=args.max_samples_per_file,
        patch_len=model.p
    )
    
    print("\n" + "=" * 50)
    print("准备验证数据...")
    print("=" * 50)
    
    eval_dataset = TimesFMTrainingDataset(
        data_paths=[],
        data_dir=train_data_dir,
        dataset_names=None,
        context_length=adjusted_context,
        horizon=args.horizon,
        stride=args.horizon,
        target_col=None,
        scale=True,
        split='val',
        train_split=0.8,
        test_split=0.1,
        max_files=args.max_files,
        max_samples_per_file=500,
        patch_len=model.p
    )
    
    print(f"\n训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset)}")
    
    if len(train_dataset) == 0:
        print("\n错误: 训练数据集为空！")
        return
    
    if len(eval_dataset) == 0:
        print("\n警告: 验证数据集为空，将只进行训练（无验证）。")
        eval_dataset = None
    
    # ========== 创建训练器并训练 ==========
    trainer = TimesFMTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        context_length=adjusted_context,
        horizon=args.horizon,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=device,
        output_dir=args.output_dir,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        lr_scheduler_type=args.lr_scheduler_type if args.lr_scheduler_type != "none" else None,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_min_lr=args.lr_scheduler_min_lr
    )
    
    trainer.train()
    
    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
