"""
TimesFM 蒸馏脚本 - 使用 TRL GKD 的适配方案

基于 chronos2_distill_gkd.py 的结构，适配 TimesFM 模型进行知识蒸馏。
本脚本提供了针对 TimesFM 的蒸馏实现。
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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

warnings.filterwarnings('ignore')

# 尝试导入 TimesFM（如果可用）
try:
    import timesfm
    # TimesFM 2.5 200M PyTorch 版本
    TimesFM_2p5_200M_torch = timesfm.TimesFM_2p5_200M_torch
    ForecastConfig = timesfm.ForecastConfig
    TIMESFM_AVAILABLE = True
except (ImportError, AttributeError) as e:
    TIMESFM_AVAILABLE = False
    print(f"警告: timesfm 模块未正确安装或导入失败: {e}")
    print("请确保已安装: pip install timesfm")


class TimesFMDistillationDataset(Dataset):
    """TimesFM 蒸馏数据集"""
    
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
        max_files: int = 100,  # 限制处理的文件数量
        max_samples_per_file: int = 1000  # 每个文件的最大样本数
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
        
        # 存储原始数据和元数据，而不是预处理后的样本
        self.data_segments = []  # 每个元素是 (ts_values, scaler, file_name)
        self.sample_offsets = []  # 每个元素是 (segment_idx, start_idx)，用于快速定位
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
            
            # 如果文件太多，限制处理数量并给出提示
            if len(csv_files) > self.max_files:
                print(f"警告: 文件数量过多 ({len(csv_files)})，将只处理前 {self.max_files} 个文件")
                print(f"如需处理所有文件，请设置 max_files 参数或使用 dataset_names 指定特定文件")
                csv_files = csv_files[:self.max_files]
            
            data_paths = [str(f) for f in csv_files]
        
        if not data_paths:
            data_paths = data_paths or []
        
        # 展开 data_paths 中的目录为文件列表
        expanded_data_paths = []
        for path in data_paths:
            if os.path.isdir(path):
                # 如果是目录，查找其中的所有 CSV 文件
                csv_files = list(Path(path).glob("*.csv"))
                if csv_files:
                    print(f"从目录 {path} 找到 {len(csv_files)} 个 CSV 文件")
                    expanded_data_paths.extend([str(f) for f in csv_files])
                else:
                    print(f"警告: 目录 {path} 中没有找到 CSV 文件，跳过")
            elif os.path.isfile(path):
                # 如果是文件，直接添加
                expanded_data_paths.append(path)
            else:
                print(f"警告: 路径 {path} 不存在，跳过")
        
        data_paths = expanded_data_paths
        
        # 计算数据划分边界
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
                
                if np.isinf(data).any():
                    print(f"  跳过 {os.path.basename(data_path)}: 包含 Inf 值")
                    continue
                
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
                
                # 使用第一个数值列作为目标（单变量预测）
                ts_values = split_data[:, 0].astype(np.float32)
                
                if len(ts_values) < context_length + horizon:
                    print(f"  跳过 {os.path.basename(data_path)}: {split} 集数据长度不足")
                    continue
                
                # 计算该数据段的样本数
                max_start_idx = len(ts_values) - context_length - horizon
                
                
                # 计算实际可用的样本数
                num_samples = (max_start_idx // stride) + 1
                if num_samples > self.max_samples_per_file:
                    num_samples = self.max_samples_per_file
                
                # 存储数据段和对应的 scaler
                segment_idx = len(self.data_segments)
                # scaler 已经在标准化步骤中存储（如果启用了标准化）
                scaler = self.scalers.get(os.path.basename(data_path), None) if self.scale else None
                self.data_segments.append({
                    'values': ts_values,
                    'scaler': scaler,
                    'file_name': os.path.basename(data_path),
                    'stride': stride,
                    'max_start_idx': max_start_idx
                })
                
                # 预计算所有样本的偏移量（用于快速定位）
                # 只生成 num_samples 个样本
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
        """根据索引动态计算样本，而不是从预存储的列表中取出"""
        if idx >= len(self.sample_offsets):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.sample_offsets)})")
        
        # 获取样本对应的数据段和起始位置
        segment_idx, start_idx = self.sample_offsets[idx]
        segment = self.data_segments[segment_idx]
        ts_values = segment['values']
        
        # 动态计算 context 和 target
        context = ts_values[start_idx:start_idx + self.context_length]
        target = ts_values[start_idx + self.context_length:start_idx + self.context_length + self.horizon]
        
        # 检查是否有 NaN 值
        if np.isnan(context).any() or np.isnan(target).any():
            # 如果包含 NaN，返回零填充（这种情况应该很少，因为在初始化时已经过滤）
            context = np.nan_to_num(context, nan=0.0)
            target = np.nan_to_num(target, nan=0.0)
        
        return {
            "target": torch.tensor(context, dtype=torch.float32),
            "future_target": torch.tensor(target, dtype=torch.float32)
        }


class SimpleTimesFMStudent(nn.Module):
    """
    简单的 TimesFM 学生模型
    使用轻量级的 Transformer 架构进行时间序列预测
    """
    
    def __init__(
        self,
        context_length: int = 96,
        horizon: int = 24,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.context_length = context_length
        self.horizon = horizon
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(1, d_model)
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.randn(context_length, d_model))
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, horizon)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            输入时间序列，形状为 [batch_size, context_length]
            
        Returns
        -------
        torch.Tensor
            预测结果，形状为 [batch_size, horizon]
        """
        # x: [batch_size, context_length]
        batch_size = x.shape[0]
        
        # 添加特征维度: [batch_size, context_length, 1]
        x = x.unsqueeze(-1)
        
        # 输入投影: [batch_size, context_length, d_model]
        x = self.input_projection(x)
        
        # 添加位置编码
        x = x + self.pos_encoder.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Transformer 编码
        encoded = self.encoder(x)  # [batch_size, context_length, d_model]
        
        # 使用最后一个时间步的输出
        last_hidden = encoded[:, -1, :]  # [batch_size, d_model]
        
        # 输出投影
        output = self.output_projection(last_hidden)  # [batch_size, horizon]
        
        return output


class TimesFMDistillationTrainer:
    """TimesFM 知识蒸馏训练器"""
    
    def __init__(
        self,
        teacher_model: Any,  # TimesFM 模型
        student_model: nn.Module,
        train_dataset: TimesFMDistillationDataset,
        eval_dataset: Optional[TimesFMDistillationDataset] = None,
        context_length: int = 96,
        horizon: int = 24,
        learning_rate: float = 5e-5,
        batch_size: int = 32,
        num_epochs: int = 10,
        temperature: float = 2.0,
        alpha: float = 0.5,  # 蒸馏损失权重
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./timesfm-distilled",
        max_grad_norm: float = 1.0,  # 梯度裁剪阈值
        gradient_accumulation_steps: int = 1  # 梯度累积步数
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.context_length = context_length
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        self.output_dir = output_dir
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # 将模型移到设备
        if hasattr(self.teacher_model, 'to'):
            self.teacher_model = self.teacher_model.to(device)
        self.student_model = self.student_model.to(device)
        
        # 设置教师模型为评估模式并冻结所有参数
        if hasattr(self.teacher_model, 'eval'):
            self.teacher_model.eval()
        if hasattr(self.teacher_model, 'parameters'):
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        print("教师模型已设置为评估模式并冻结所有参数")
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            eps=1e-8
        )
        
        # 数据加载器
        pin_memory = device == "cpu"
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
    
    def _teacher_predict(self, context_batch: torch.Tensor) -> torch.Tensor:
        """
        使用教师模型（TimesFM）进行预测（分批处理以节省内存）
        
        TimesFM API:
        point_forecast, quantile_forecast = model.forecast(
            horizon=horizon,
            inputs=[numpy_array1, numpy_array2, ...]
        )
        
        Parameters
        ----------
        context_batch : torch.Tensor
            输入上下文数据，形状为 [batch_size, context_length]
            
        Returns
        -------
        torch.Tensor
            教师模型预测结果，形状为 [batch_size, horizon]
        """
        batch_size = context_batch.shape[0]
        
        # 将 tensor 转换为 numpy array 列表
        context_np = context_batch.cpu().numpy()
        
        # 分批处理以减少内存使用（每批最多 16 个样本）
        teacher_batch_size = min(16, batch_size)
        all_predictions = []
        
        try:
            for i in range(0, batch_size, teacher_batch_size):
                end_idx = min(i + teacher_batch_size, batch_size)
                batch_inputs = [context_np[j] for j in range(i, end_idx)]
                
                # 使用 TimesFM 的 forecast 方法
                # forecast 返回 (point_forecast, quantile_forecast)
                # 添加超时保护：如果单个批次预测时间过长，记录警告
                import time
                start_time = time.time()
                point_forecast, quantile_forecast = self.teacher_model.forecast(
                    horizon=self.horizon,
                    inputs=batch_inputs
                )
                elapsed = time.time() - start_time
                if elapsed > 5.0:  # 如果单个批次预测超过5秒，记录警告
                    print(f"  警告: 教师模型预测批次 [{i}:{end_idx}] 耗时 {elapsed:.2f} 秒")
                
                # 使用点预测（point_forecast）
                if isinstance(point_forecast, np.ndarray):
                    batch_logits = torch.tensor(point_forecast, dtype=torch.float32)
                elif torch.is_tensor(point_forecast):
                    batch_logits = point_forecast.float()
                else:
                    # 如果返回格式不同，尝试转换
                    batch_logits = torch.tensor(np.array(point_forecast), dtype=torch.float32)
                
                all_predictions.append(batch_logits)
                
                # 清理 GPU 缓存
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            
            # 合并所有批次的预测结果
            teacher_logits = torch.cat(all_predictions, dim=0).to(self.device)
            
            # 确保形状正确
            if teacher_logits.shape[0] != batch_size:
                raise ValueError(f"预测结果批次大小不匹配: 期望 {batch_size}, 得到 {teacher_logits.shape[0]}")
            if teacher_logits.shape[1] != self.horizon:
                # 如果预测长度不匹配，截断或填充
                if teacher_logits.shape[1] > self.horizon:
                    teacher_logits = teacher_logits[:, :self.horizon]
                else:
                    padding = torch.zeros(
                        batch_size,
                        self.horizon - teacher_logits.shape[1],
                        device=self.device
                    )
                    teacher_logits = torch.cat([teacher_logits, padding], dim=1)
            
            return teacher_logits
            
        except Exception as e:
            print(f"警告: 教师模型预测失败: {e}")
            import traceback
            traceback.print_exc()
            # 使用零填充作为后备
            return torch.zeros(batch_size, self.horizon, dtype=torch.float32).to(self.device)
    
    def train_epoch(self, epoch: int):
        """训练一个 epoch"""
        self.student_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            if (batch_idx + 1) % 10 == 0:
                print(f"  [Batch {batch_idx + 1}] 开始加载数据...")
            context = batch["target"].to(self.device)  # [batch_size, context_length]
            future_target = batch["future_target"].to(self.device)  # [batch_size, horizon]
            
            batch_size = context.shape[0]
            if (batch_idx + 1) % 10 == 0:
                print(f"  [Batch {batch_idx + 1}] 数据加载完成，batch_size={batch_size}")
            
            # 教师模型预测（不计算梯度）
            with torch.no_grad():
                try:
                    if (batch_idx + 1) % 10 == 0:
                        print(f"  [Batch {batch_idx + 1}] 开始教师模型预测...")
                    teacher_logits = self._teacher_predict(context)
                    if (batch_idx + 1) % 10 == 0:
                        print(f"  [Batch {batch_idx + 1}] 教师模型预测完成")
                    # 清理 GPU 缓存以释放内存
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"警告: 教师模型预测失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 学生模型前向传播
            # 只在梯度累积开始时清零梯度
            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            try:
                # 调用学生模型的 forward 方法
                # SimpleTimesFMStudent 直接接受 context 并输出预测
                student_logits = self.student_model(context)  # [batch_size, horizon]
                    
            except Exception as e:
                print(f"警告: 学生模型前向传播失败: {e}")
                import traceback
                traceback.print_exc()
                # 清理 GPU 缓存
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                continue
            
            # 确保 teacher_logits 和 student_logits 形状匹配
            if teacher_logits.shape != student_logits.shape:
                min_len = min(teacher_logits.shape[1], student_logits.shape[1])
                teacher_logits = teacher_logits[:, :min_len]
                student_logits = student_logits[:, :min_len]
                future_target = future_target[:, :min_len]
            
            # 检查是否有 NaN 或 Inf 值
            if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                print(f"警告: Batch {batch_idx + 1} 学生模型输出包含 NaN/Inf，跳过")
                continue
            
            if torch.isnan(teacher_logits).any() or torch.isinf(teacher_logits).any():
                print(f"警告: Batch {batch_idx + 1} 教师模型输出包含 NaN/Inf，跳过")
                continue
            
            # 计算损失（使用 MSE，因为这是回归任务）
            mse_loss = F.mse_loss(student_logits, future_target)
            distillation_loss = F.mse_loss(student_logits, teacher_logits)
            
            # 检查损失是否为 NaN 或 Inf
            if torch.isnan(mse_loss) or torch.isinf(mse_loss) or torch.isnan(distillation_loss) or torch.isinf(distillation_loss):
                print(f"警告: Batch {batch_idx + 1} 损失包含 NaN/Inf，跳过")
                continue
            
            # 组合损失
            loss = self.alpha * distillation_loss + (1 - self.alpha) * mse_loss
            
            # 保存原始损失值用于打印和统计
            original_loss = loss.item()
            
            # 限制损失值，防止数值爆炸
            if original_loss > 1e6:
                print(f"警告: Batch {batch_idx + 1} 损失过大 ({original_loss:.2f})，跳过")
                continue
            
            # 反向传播（除以累积步数以平均梯度）
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            # 只在累积步数达到时更新参数
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 检查梯度是否包含 NaN
                has_nan_grad = False
                for param in self.student_model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break
                
                if has_nan_grad:
                    print(f"警告: Batch {batch_idx + 1} 梯度包含 NaN/Inf，跳过更新")
                    self.optimizer.zero_grad()
                    # 清理 GPU 缓存
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    continue
                
                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.max_grad_norm)
                
                # 更新参数
                self.optimizer.step()
                
                # 清理 GPU 缓存
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                if (batch_idx + 1) % 100 == 0:
                    print(f"  梯度范数: {grad_norm.item():.4f}")
            
            # 使用原始损失值进行统计
            total_loss += original_loss
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(self.train_loader)}, "
                      f"Loss: {original_loss:.4f}, MSE: {mse_loss.item():.4f}, Distill: {distillation_loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def evaluate(self):
        """评估模型"""
        if self.eval_loader is None:
            return None
        
        self.student_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                context = batch["target"].to(self.device)
                future_target = batch["future_target"].to(self.device)
                
                try:
                    # 学生模型直接预测
                    student_logits = self.student_model(context)  # [batch_size, horizon]
                    
                    min_len = min(student_logits.shape[1], future_target.shape[1])
                    loss = F.mse_loss(student_logits[:, :min_len], future_target[:, :min_len])
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
        print("开始 TimesFM 蒸馏训练")
        print("=" * 50)
        
        best_eval_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)
            
            train_loss = self.train_epoch(epoch)
            print(f"训练损失: {train_loss:.4f}")
            
            if self.eval_loader:
                eval_loss = self.evaluate()
                if eval_loss is not None:
                    print(f"验证损失: {eval_loss:.4f}")
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self.save_model(f"best_model_epoch_{epoch + 1}")
                        print(f"保存最佳模型 (验证损失: {eval_loss:.4f})")
        
        self.save_model("final_model")
        print("\n训练完成！")
    
    def save_model(self, model_name: str):
        """保存模型"""
        save_path = os.path.join(self.output_dir, model_name)
        os.makedirs(save_path, exist_ok=True)
        # 保存 PyTorch 模型
        torch.save({
            'model_state_dict': self.student_model.state_dict(),
            'context_length': self.context_length,
            'horizon': self.horizon,
        }, os.path.join(save_path, 'pytorch_model.bin'))
        print(f"模型已保存到: {save_path}")


def create_student_model(
    context_length: int = 96,
    horizon: int = 24,
    student_config_overrides: Optional[Dict[str, Any]] = None
) -> SimpleTimesFMStudent:
    """
    创建学生模型（更小的架构）
    
    Parameters
    ----------
    context_length : int
        上下文长度
    horizon : int
        预测长度
    student_config_overrides : Optional[Dict[str, Any]]
        学生模型配置覆盖参数，例如：
        {
            "d_model": 256,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 512,
            "dropout": 0.1
        }
        
    Returns
    -------
    SimpleTimesFMStudent
        学生模型
    """
    # 默认配置
    config = {
        "context_length": context_length,
        "horizon": horizon,
        "d_model": 256,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 512,
        "dropout": 0.1
    }
    
    # 应用配置覆盖
    if student_config_overrides:
        config.update(student_config_overrides)
    
    student_model = SimpleTimesFMStudent(**config)
    return student_model


def main():
    """主函数"""
    
    if not TIMESFM_AVAILABLE:
        print("错误: timesfm 模块未安装。请先安装: pip install timesfm")
        return
    
    # ========== 配置参数 ==========
    teacher_model_id = "google/timesfm-2.5-200m-pytorch"  # TimesFM 模型 ID
    output_dir = "./timesfm-distilled"
    
    # 数据配置
    # 尝试多个可能的数据路径
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
        print("错误: 未找到数据目录。请检查以下路径之一是否存在:")
        for data_dir in possible_data_dirs:
            print(f"  - {data_dir}")
        print("\n或者修改代码中的 train_data_dir 变量指向正确的数据目录。")
        return
    
    train_dataset_names = None  # None 表示加载目录下所有 CSV 文件
    train_data_paths = []
    
    # 验证数据（使用相同的目录）
    eval_data_paths = [train_data_dir]
    eval_dataset_names = None
    
    context_length = 720
    horizon = 96
    stride = 2  
    target_col = None
    max_files = 50  # 限制处理的文件数量，避免处理时间过长
    
    # 学生模型配置
    student_config_overrides = {
        "d_model": 256,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 512,
        "dropout": 0.1
    }
    
    # 训练配置
    learning_rate = 1e-5
    batch_size = 64  # 减小批次大小以节省内存
    gradient_accumulation_steps = 8  # 梯度累积，等效 batch_size = 64 * 8 = 5126
    num_epochs = 10
    temperature = 2.0
    alpha = 0.5
    max_grad_norm = 1.0
    
    # ========== 加载教师模型 ==========
    print("=" * 50)
    print("加载教师模型 (TimesFM)...")
    print("=" * 50)
    try:
        # 加载 TimesFM 2.5 200M PyTorch 模型
        teacher_model = TimesFM_2p5_200M_torch.from_pretrained(
            teacher_model_id,
            torch_compile=True
        )
        print(f"教师模型加载完成: {teacher_model_id}")
        
        # 配置模型（compile）
        print("配置模型...")
        teacher_model.compile(
            ForecastConfig(
                max_context=max(context_length, 1024),  # 至少 1024
                max_horizon=max(horizon, 256),  # 至少 256
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        print("模型配置完成")
        
        # 计算参数量（如果可能）
        if hasattr(teacher_model, 'parameters'):
            teacher_params = sum(p.numel() for p in teacher_model.parameters())
            print(f"教师模型参数量: {teacher_params:,}")
    except Exception as e:
        print(f"加载教师模型失败: {e}")
        import traceback
        traceback.print_exc()
        print("请检查模型 ID 是否正确，以及 timesfm 库是否已正确安装。")
        return
    
    # ========== 创建学生模型 ==========
    print("\n" + "=" * 50)
    print("创建学生模型...")
    print("=" * 50)
    student_model = create_student_model(
        context_length=context_length,
        horizon=horizon,
        student_config_overrides=student_config_overrides
    )
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"学生模型参数量: {student_params:,}")
    if hasattr(teacher_model, 'parameters'):
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        print(f"压缩比: {student_params / teacher_params:.2%}")
    
    # ========== 准备数据 ==========
    print("\n" + "=" * 50)
    print("准备训练数据...")
    print("=" * 50)
    
    train_dataset = TimesFMDistillationDataset(
        data_paths=train_data_paths,
        data_dir=train_data_dir,
        dataset_names=train_dataset_names,
        context_length=context_length,
        horizon=horizon,
        stride=stride,
        target_col=target_col,
        scale=True,
        split='train',
        train_split=0.8,
        test_split=0.1,
        max_files=max_files,
        max_samples_per_file=1000
    )
    
    print("\n" + "=" * 50)
    print("准备验证数据...")
    print("=" * 50)
    eval_dataset = TimesFMDistillationDataset(
        data_paths=eval_data_paths,
        dataset_names=eval_dataset_names,
        context_length=context_length,
        horizon=horizon,
        stride=horizon,
        target_col=target_col,
        scale=True,
        split='val',
        train_split=0.8,
        test_split=0.1,
        max_files=max_files,
        max_samples_per_file=500  # 验证集使用更少的样本
    )
    
    print(f"\n训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset)}")
    
    # 检查数据集是否为空
    if len(train_dataset) == 0:
        print("\n错误: 训练数据集为空！")
        print("可能的原因:")
        print("  1. 数据目录中没有 CSV 文件")
        print("  2. CSV 文件格式不正确")
        print("  3. 数据长度不足（需要至少 context_length + horizon 个数据点）")
        print(f"  4. 数据目录路径不正确: {train_data_dir}")
        print("\n请检查数据目录和文件格式。")
        return
    
    if len(eval_dataset) == 0:
        print("\n警告: 验证数据集为空，将只进行训练（无验证）。")
        eval_dataset = None
    
    # ========== 创建训练器并训练 ==========
    trainer = TimesFMDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        context_length=context_length,
        horizon=horizon,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        temperature=temperature,
        alpha=alpha,
        output_dir=output_dir,
        max_grad_norm=max_grad_norm,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    trainer.train()
    
    print("\n" + "=" * 50)
    print("蒸馏完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()

