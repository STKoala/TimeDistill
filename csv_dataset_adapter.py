#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
CSV 数据集适配器
将 CSV 文件转换为 TimeMoE 可以使用的数据集格式
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

# 添加 Time-MoE 路径
import sys
timemoe_path = Path(__file__).parent.parent / "Time-MoE"
if timemoe_path.exists():
    sys.path.insert(0, str(timemoe_path))

try:
    from time_moe.datasets.ts_dataset import TimeSeriesDataset
    TIMEMOE_AVAILABLE = True
except ImportError:
    TIMEMOE_AVAILABLE = False
    print("警告: Time-MoE 模块未找到")


class CSVTimeSeriesDataset(TimeSeriesDataset):
    """CSV 时间序列数据集，兼容 TimeMoE 格式"""
    
    def __init__(self, data_folder: str, normalization_method=None, time_col_name: str = 'date'):
        """
        Args:
            data_folder: CSV 文件所在的文件夹路径
            normalization_method: 归一化方法（None, 'zero', 'max'）
            time_col_name: 时间列名称（会被忽略）
        """
        if not TIMEMOE_AVAILABLE:
            raise ImportError("Time-MoE 模块未找到，无法使用 CSVTimeSeriesDataset")
        
        self.data_folder = data_folder
        self.time_col_name = time_col_name
        self.normalization_method = normalization_method
        self.sequences = []
        self.num_tokens = None
        
        # 设置归一化方法
        if normalization_method is None:
            self.norm_func = None
        elif isinstance(normalization_method, str):
            if normalization_method.lower() == 'max':
                self.norm_func = self._max_scaler
            elif normalization_method.lower() == 'zero':
                self.norm_func = self._zero_scaler
            else:
                raise ValueError(f'Unknown normalization method: {normalization_method}')
        else:
            self.norm_func = normalization_method
        
        # 加载所有 CSV 文件
        self._load_csv_files()
        
        if len(self.sequences) == 0:
            raise ValueError(f"没有找到有效的 CSV 文件或数据序列: {data_folder}")
    
    def _load_csv_files(self):
        """加载文件夹中的所有 CSV 文件"""
        csv_files = list(Path(self.data_folder).glob("*.csv"))
        
        print(f"找到 {len(csv_files)} 个 CSV 文件")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # 排除时间列
                exclude_cols = ['date', 'timestamp', 'time', 'id', 'item_id', 'time_idx', self.time_col_name]
                numeric_cols = [col for col in df.columns 
                               if col.lower() not in [c.lower() for c in exclude_cols]
                               and pd.api.types.is_numeric_dtype(df[col])]
                
                if len(numeric_cols) == 0:
                    # 如果没有数值列，尝试使用所有列（除了时间列）
                    numeric_cols = [col for col in df.columns if col.lower() not in [c.lower() for c in exclude_cols]]
                    if len(numeric_cols) == 0:
                        print(f"  跳过 {csv_file.name}: 没有数值列")
                        continue
                
                # 使用第一个数值列（单变量预测）
                ts_values = df[numeric_cols[0]].values.astype(np.float32)
                
                # 检查数据有效性
                if len(ts_values) == 0:
                    print(f"  跳过 {csv_file.name}: 数据为空")
                    continue
                
                # 移除 NaN 和 Inf
                ts_values = ts_values[~np.isnan(ts_values)]
                ts_values = ts_values[~np.isinf(ts_values)]
                
                if len(ts_values) == 0:
                    print(f"  跳过 {csv_file.name}: 移除 NaN/Inf 后数据为空")
                    continue
                
                # 应用归一化
                if self.norm_func is not None:
                    ts_values = self.norm_func(ts_values)
                
                # 每个 CSV 文件作为一个序列
                self.sequences.append(ts_values)
                
            except Exception as e:
                print(f"  跳过 {csv_file.name}: 加载失败 - {e}")
                continue
        
        print(f"成功加载 {len(self.sequences)} 个时间序列")
    
    def _zero_scaler(self, seq):
        """零均值归一化"""
        if not isinstance(seq, np.ndarray):
            seq = np.array(seq)
        origin_dtype = seq.dtype
        mean_val = seq.mean()
        std_val = seq.std()
        if std_val == 0:
            normed_seq = seq - mean_val
        else:
            normed_seq = (seq - mean_val) / std_val
        return normed_seq.astype(origin_dtype)
    
    def _max_scaler(self, seq):
        """最大绝对值归一化"""
        if not isinstance(seq, np.ndarray):
            seq = np.array(seq)
        origin_dtype = seq.dtype
        max_val = np.abs(seq).max()
        if max_val == 0:
            normed_seq = seq
        else:
            normed_seq = seq / max_val
        return normed_seq.astype(origin_dtype)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, seq_idx):
        if seq_idx < 0 or seq_idx >= len(self.sequences):
            raise IndexError(f"Index {seq_idx} out of range [0, {len(self.sequences)})")
        return self.sequences[seq_idx]
    
    def get_sequence_length_by_idx(self, seq_idx):
        if seq_idx < 0 or seq_idx >= len(self.sequences):
            raise IndexError(f"Index {seq_idx} out of range [0, {len(self.sequences)})")
        return len(self.sequences[seq_idx])
    
    def get_num_tokens(self):
        if self.num_tokens is None:
            self.num_tokens = sum([len(seq) for seq in self.sequences])
        return self.num_tokens

