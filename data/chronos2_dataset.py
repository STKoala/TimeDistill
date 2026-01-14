"""
Chronos-2 蒸馏数据集（参考 LightGTS 的处理方式）
"""

import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Optional
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class Chronos2DistillationDataset(Dataset):
    """Chronos-2 蒸馏数据集（参考 LightGTS 的处理方式）"""
    
    def __init__(
        self,
        data_paths: List[str] = None,
        context_length: int = 96,
        horizon: int = 24,
        stride: int = 1,
        target_col: str = None,
        data_dir: str = None,  # 数据目录
        dataset_names: List[str] = None,  # 数据集名称列表（如 ['ETTh2', 'kdd_cup_2018_dataset_without_missing_values(0)']）
        scale: bool = True,  # 是否标准化
        train_split: float = 0.8,  # 训练集比例
        test_split: float = 0.1,  # 测试集比例
        split: str = 'train',  # 'train', 'val', 'test'
        time_col_name: str = 'date'  # 时间列名称
    ):
        self.context_length = context_length
        self.horizon = horizon
        self.scale = scale
        self.split = split
        self.time_col_name = time_col_name
        self.samples = []
        self.scalers = {}  # 存储每个数据集的 scaler
        
        # 如果提供了数据集名称列表，从目录中查找对应的 CSV 文件
        if dataset_names and data_dir:
            print(f"从目录加载指定数据集: {data_dir}")
            data_paths = []
            for name in dataset_names:
                # 尝试多种可能的文件名格式
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
            data_paths = [str(f) for f in csv_files]
        
        # 如果没有提供任何路径，使用 data_paths
        if not data_paths:
            data_paths = data_paths or []
        
        # 计算数据划分边界
        for data_path in data_paths:
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
                
                # 只使用数值列
                df_data = df_data.select_dtypes(include=[np.number])
                
                if len(df_data.columns) == 0:
                    print(f"  跳过 {os.path.basename(data_path)}: 没有数值列")
                    continue
                
                # 提取时间序列数据
                data = df_data.values.astype(np.float32)
                
                # 检查数据有效性
                if len(data) == 0:
                    print(f"  跳过 {os.path.basename(data_path)}: 没有有效数据")
                    continue
                
                # 检查是否有异常值
                if np.isinf(data).any():
                    print(f"  跳过 {os.path.basename(data_path)}: 包含 Inf 值")
                    continue
                
                # 数据划分（参考 LightGTS）
                num_train = int(len(data) * train_split)
                num_test = int(len(data) * test_split)
                num_val = len(data) - num_train - num_test
                
                border1s = [0, num_train - context_length, len(data) - num_test - context_length]
                border2s = [num_train, num_train + num_val, len(data)]
                
                split_map = {'train': 0, 'val': 1, 'test': 2}
                split_idx = split_map.get(split, 0)
                border1 = border1s[split_idx]
                border2 = border2s[split_idx]
                
                # 提取对应 split 的数据
                split_data = data[border1:border2]
                
                if len(split_data) == 0:
                    print(f"  跳过 {os.path.basename(data_path)}: {split} 集为空")
                    continue
                
                # 数据标准化（参考 LightGTS，使用训练集拟合 scaler）
                if self.scale:
                    train_data = data[border1s[0]:border2s[0]]
                    scaler = StandardScaler()
                    scaler.fit(train_data)
                    split_data = scaler.transform(split_data)
                    # 存储 scaler（如果需要）
                    self.scalers[os.path.basename(data_path)] = scaler
                
                # 使用第一个数值列作为目标（单变量预测）
                ts_values = split_data[:, 0].astype(np.float32)
                
                # 确保有足够的数据
                if len(ts_values) < context_length + horizon:
                    print(f"  跳过 {os.path.basename(data_path)}: {split} 集数据长度不足 ({len(ts_values)} < {context_length + horizon})")
                    continue
                
                # 使用滑动窗口创建样本
                max_start_idx = len(ts_values) - context_length - horizon
                
                file_samples = 0
                for start_idx in range(0, max_start_idx + 1, stride):
                    context = ts_values[start_idx:start_idx + context_length]
                    target = ts_values[start_idx + context_length:start_idx + context_length + horizon]
                    
                    # 检查是否有 NaN 值
                    if np.isnan(context).any() or np.isnan(target).any():
                        continue
                    
                    self.samples.append({
                        "target": torch.tensor(context, dtype=torch.float32),
                        "future_target": torch.tensor(target, dtype=torch.float32)
                    })
                    file_samples += 1
                
                if file_samples > 0:
                    print(f"  {os.path.basename(data_path)} ({split}): 添加了 {file_samples} 个样本 (目标列: {target_col_actual})")
                    
            except Exception as e:
                print(f"  处理 {os.path.basename(data_path)} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n共准备 {len(self.samples)} 个 {split} 样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

