"""
Chronos-2 蒸馏脚本 - 使用 TRL GKD 的适配方案

由于 Chronos-2 使用自定义架构（非标准 Transformer），直接使用 TRL GKD 可能不兼容。
本脚本提供了一个可行的适配方案：

方案 1：使用 Chronos-2 的 fit 方法 + 手动蒸馏损失（推荐）
方案 2：将 Chronos-2 模型包装为兼容 GKD 的格式（实验性）

本脚本实现方案 1，这是最稳定和可行的方案。
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 限制只使用第一个GPU，避免多GPU分布导致设备不匹配问题
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import TrainingArguments, Trainer
from chronos import Chronos2Pipeline, Chronos2Model
from chronos.chronos2 import Chronos2ForecastingConfig, Chronos2CoreConfig
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class DTWLoss(nn.Module):
    """
    可微分的DTW（Dynamic Time Warping）损失函数
    
    DTW用于计算两个时间序列之间的对齐距离，对时间偏移更加鲁棒。
    本实现使用soft-DTW的思想，通过可微分的动态规划实现。
    """
    
    def __init__(self, gamma: float = 1.0, use_fast_approx: bool = True):
        """
        Parameters
        ----------
        gamma : float
            Soft-DTW的温度参数，gamma越小越接近硬DTW，gamma越大越平滑
        use_fast_approx : bool
            如果True，使用快速近似实现（避免复杂的动态规划）
            如果False，使用标准的soft-DTW实现（较慢但更精确）
        """
        super().__init__()
        self.gamma = gamma
        self.use_fast_approx = use_fast_approx
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算DTW损失
        
        Parameters
        ----------
        pred : torch.Tensor
            预测序列，形状为 (batch_size, seq_len) 或 (batch_size, seq_len, 1)
        target : torch.Tensor
            目标序列，形状为 (batch_size, seq_len) 或 (batch_size, seq_len, 1)
            
        Returns
        -------
        torch.Tensor
            DTW损失值（标量）
        """
        # 确保输入是2D的 (batch_size, seq_len)
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        if target.dim() == 3:
            target = target.squeeze(-1)
        
        # 确保pred和target长度相同（如果不同，截断到最小长度）
        min_len = min(pred.shape[1], target.shape[1])
        pred = pred[:, :min_len]
        target = target[:, :min_len]
        
        batch_size = pred.shape[0]
        seq_len = pred.shape[1]
        
        # 批量计算距离矩阵
        # pred: (batch_size, seq_len) -> (batch_size, seq_len, 1)
        # target: (batch_size, seq_len) -> (batch_size, 1, seq_len)
        pred_expanded = pred.unsqueeze(2)  # (batch_size, seq_len, 1)
        target_expanded = target.unsqueeze(1)  # (batch_size, 1, seq_len)
        distance_matrix = (pred_expanded - target_expanded) ** 2  # (batch_size, seq_len, seq_len)
        
        # 批量计算DTW损失
        if self.use_fast_approx:
            # 使用快速近似：计算加权平均距离
            # 这避免了复杂的动态规划，同时保持可微分性
            # 对于时间序列，使用soft-alignment权重
            weights = torch.softmax(-distance_matrix / self.gamma, dim=2)  # (batch_size, n, m)
            # 计算加权距离
            weighted_dist = torch.sum(distance_matrix * weights, dim=(1, 2))  # (batch_size,)
            # 归一化
            losses = weighted_dist / (seq_len * seq_len)
        else:
            # 使用标准soft-DTW实现
            losses = []
            for i in range(batch_size):
                dtw_loss = self._soft_dtw(distance_matrix[i], self.gamma)
                losses.append(dtw_loss)
            losses = torch.stack(losses)
        
        # 返回平均损失
        return losses.mean()
    
    def _soft_dtw(self, distance_matrix: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        计算soft-DTW距离（优化实现，使用预分配矩阵避免频繁操作）
        
        注意：为了性能，这里使用一个简化的实现。R矩阵使用detach来避免梯度问题，
        但最终结果通过distance_matrix保持梯度连接。
        
        Parameters
        ----------
        distance_matrix : torch.Tensor
            距离矩阵，形状为 (n, m)
        gamma : float
            温度参数（必须>0）
            
        Returns
        -------
        torch.Tensor
            soft-DTW距离
        """
        n, m = distance_matrix.shape
        
        # 如果gamma太小，使用一个最小值避免数值不稳定
        if gamma <= 0:
            gamma = 1e-6
        
        # 使用预分配的矩阵，避免频繁的torch.cat操作
        large_value = 1e6
        
        # 初始化R矩阵（使用numpy风格的数组，避免梯度问题）
        # 使用torch.zeros初始化，然后填充大值
        R = torch.full(
            (n + 1, m + 1),
            large_value,
            device=distance_matrix.device,
            dtype=distance_matrix.dtype
        )
        R[0, 0] = 0.0
        
        # 逐行计算（R不需要梯度，但计算过程中使用distance_matrix保持梯度）
        # 使用with torch.no_grad()来避免R的梯度计算
        with torch.no_grad():
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    # 获取三个方向的值
                    diag = R[i - 1, j - 1]
                    up = R[i - 1, j]
                    left = R[i, j - 1]
                    
                    # 计算三个方向的soft-min（在no_grad上下文中）
                    r_prev = torch.stack([diag, up, left])
                    r_prev_neg = -r_prev / gamma
                    r_max = torch.max(r_prev_neg)
                    exp_values = torch.exp(r_prev_neg - r_max)
                    log_sum = torch.log(torch.sum(exp_values) + 1e-10)
                    r_min = -gamma * (r_max + log_sum)
                    
                    # 计算新的累积距离（在no_grad中，使用detach的distance_matrix）
                    # 注意：这里我们使用distance_matrix的值，但不保持梯度
                    dist_val = distance_matrix[i - 1, j - 1].detach()
                    new_val = dist_val + r_min
                    R[i, j] = new_val
        
        # 重新计算最后一步，保持梯度连接
        # 这是关键：我们需要重新计算最后一步，确保梯度能够通过distance_matrix传播
        if n > 0 and m > 0:
            # 获取最后一步的三个方向的值（从R中获取，R已计算完成）
            diag = R[n - 1, m - 1]
            up = R[n - 1, m]
            left = R[n, m - 1]
            
            # 重新计算soft-min（这次保持梯度）
            r_prev = torch.stack([diag, up, left])
            r_prev_neg = -r_prev / gamma
            r_max = torch.max(r_prev_neg)
            exp_values = torch.exp(r_prev_neg - r_max)
            log_sum = torch.log(torch.sum(exp_values) + 1e-10)
            r_min = -gamma * (r_max + log_sum)
            
            # 最终结果（保持梯度连接，通过distance_matrix）
            result = distance_matrix[n - 1, m - 1] + r_min
        else:
            # 如果n或m为0，直接返回0
            result = torch.tensor(0.0, device=distance_matrix.device, dtype=distance_matrix.dtype)
        
        # 如果结果异常，使用简单的MSE作为fallback（保持梯度连接）
        if not torch.isfinite(result):
            result = torch.mean(distance_matrix)
        
        return result


class FeatureRegressor(nn.Module):
    """特征回归器，用于将教师模型的特征维度转换为学生模型的维度"""
    
    def __init__(self, teacher_dim: int, student_dim: int):
        """
        Parameters
        ----------
        teacher_dim : int
            教师模型的隐藏层维度
        student_dim : int
            学生模型的隐藏层维度
        """
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(teacher_dim, student_dim),
            nn.LayerNorm(student_dim),
            nn.GELU(),
            nn.Linear(student_dim, student_dim)
        )
    
    def forward(self, teacher_features: torch.Tensor) -> torch.Tensor:
        """
        将教师特征转换为学生维度
        
        Parameters
        ----------
        teacher_features : torch.Tensor
            教师模型特征，形状为 (batch_size, seq_len, teacher_dim)
            
        Returns
        -------
        torch.Tensor
            转换后的特征，形状为 (batch_size, seq_len, student_dim)
        """
        return self.regressor(teacher_features)


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


def move_model_to_device(model: nn.Module, device: str) -> nn.Module:
    """
    将模型移动到指定设备
    
    注意：如果模型使用了accelerate hooks，不能直接移动。
    在这种情况下，应该使用accelerate的API来管理设备，或者禁用accelerate。
    
    Parameters
    ----------
    model : nn.Module
        要移动的模型
    device : str
        目标设备（如 "cuda:0" 或 "cpu"）
    
    Returns
    -------
    nn.Module
        移动后的模型
    """
    # 检查模型是否使用了accelerate hooks
    # 如果使用了accelerate，直接调用.to()会触发警告
    # 但在我们的场景中，我们在加载时已经设置了device_map=None
    # 所以应该可以安全地移动
    
    try:
        model = model.to(device)
    except RuntimeError as e:
        if "accelerate hooks" in str(e).lower() or "dispatched" in str(e).lower():
            raise RuntimeError(
                f"模型使用了accelerate hooks，无法直接移动。"
                f"请确保在加载模型时设置device_map=None以禁用accelerate的自动设备映射。"
            ) from e
        raise
    
    return model


class Chronos2DistillationTrainer:
    """Chronos-2 知识蒸馏训练器"""
    
    def __init__(
        self,
        teacher_pipeline: Chronos2Pipeline,
        student_model: Chronos2Model,
        train_dataset: Chronos2DistillationDataset,
        eval_dataset: Optional[Chronos2DistillationDataset] = None,
        context_length: int = 96,
        horizon: int = 24,
        learning_rate: float = 5e-5,
        batch_size: int = 32,
        num_epochs: int = 10,
        temperature: float = 2.0,
        alpha: float = 0.5,  # 蒸馏损失权重
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./chronos-2-distilled",
        max_grad_norm: float = 1.0,  # 梯度裁剪阈值
        dtw_gamma: float = 1.0,  # DTW损失的温度参数
        dtw_weight: float = 0.2  # DTW损失的权重
    ):
        self.teacher_pipeline = teacher_pipeline
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
        self.dtw_gamma = dtw_gamma
        self.dtw_weight = dtw_weight
        
        # 初始化DTW损失函数（使用快速近似以提高性能）
        self.dtw_loss_fn = DTWLoss(gamma=dtw_gamma, use_fast_approx=True).to(device)
        print(f"初始化DTW损失函数: gamma={dtw_gamma}, weight={dtw_weight}, use_fast_approx=True")
        
        # 将模型移到设备（使用辅助函数确保所有参数都在同一设备上）
        print(f"将教师模型移动到设备: {device}")
        self.teacher_pipeline.model = move_model_to_device(self.teacher_pipeline.model, device)
        print(f"将学生模型移动到设备: {device}")
        self.student_model = move_model_to_device(self.student_model, device)
        
        # 设置教师模型为评估模式并冻结所有参数
        self.teacher_pipeline.model.eval()
        for param in self.teacher_pipeline.model.parameters():
            param.requires_grad = False
        print("教师模型已设置为评估模式并冻结所有参数")
        
        # 创建特征回归器（用于维度对齐）
        teacher_dim = teacher_pipeline.model.config.d_model
        student_dim = student_model.config.d_model
        self.feature_regressor_first = FeatureRegressor(teacher_dim, student_dim).to(device)
        self.feature_regressor_last = FeatureRegressor(teacher_dim, student_dim).to(device)
        print(f"创建特征回归器: 教师维度={teacher_dim}, 学生维度={student_dim}")
        
        # 用于存储hook提取的特征
        self.teacher_features = {}
        self.student_features = {}
        
        # 初始化handles列表
        self.teacher_handles = []
        self.student_handles = []
        
        # 注册hook来提取特征
        self._register_hooks()
        
        # 优化器（包含学生模型和regressor的参数）
        optimizer_params = list(self.student_model.parameters()) + \
                          list(self.feature_regressor_first.parameters()) + \
                          list(self.feature_regressor_last.parameters())
        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=learning_rate,
            weight_decay=1e-4,  # 添加权重衰减
            eps=1e-8  # 数值稳定性
        )
        
        # 数据加载器（如果使用 GPU，pin_memory 应该为 False，因为数据已经在 GPU 上）
        pin_memory = device == "cpu"
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0 if device != "cpu" else 4,  # GPU 时使用 0 workers 避免序列化问题
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
    
    def _register_hooks(self):
        """注册hook来提取教师和学生模型的注意力层特征"""
        # 清除之前的hook
        self.teacher_handles = []
        self.student_handles = []
        
        # 获取教师模型的编码器层数
        teacher_model = self.teacher_pipeline.model
        teacher_num_layers = len(teacher_model.encoder.block)
        
        # 获取学生模型的编码器层数
        student_num_layers = len(self.student_model.encoder.block)
        
        # 教师模型的第一层（索引0）和最后一层（索引teacher_num_layers-1）
        teacher_first_layer = teacher_model.encoder.block[0]
        teacher_last_layer = teacher_model.encoder.block[teacher_num_layers - 1]
        
        # 学生模型的第一层（索引0）和最后一层（索引student_num_layers-1）
        student_first_layer = self.student_model.encoder.block[0]
        student_last_layer = self.student_model.encoder.block[student_num_layers - 1]
        
        # 定义hook函数
        def get_teacher_first_hook(name):
            def hook(module, input, output):
                # output是Chronos2EncoderBlockOutput，包含hidden_states
                if hasattr(output, 'hidden_states'):
                    self.teacher_features['first'] = output.hidden_states
                elif isinstance(output, tuple):
                    self.teacher_features['first'] = output[0]
                else:
                    self.teacher_features['first'] = output
            return hook
        
        def get_teacher_last_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'hidden_states'):
                    self.teacher_features['last'] = output.hidden_states
                elif isinstance(output, tuple):
                    self.teacher_features['last'] = output[0]
                else:
                    self.teacher_features['last'] = output
            return hook
        
        def get_student_first_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'hidden_states'):
                    self.student_features['first'] = output.hidden_states
                elif isinstance(output, tuple):
                    self.student_features['first'] = output[0]
                else:
                    self.student_features['first'] = output
            return hook
        
        def get_student_last_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'hidden_states'):
                    self.student_features['last'] = output.hidden_states
                elif isinstance(output, tuple):
                    self.student_features['last'] = output[0]
                else:
                    self.student_features['last'] = output
            return hook
        
        # 注册教师模型的hook
        teacher_first_handle = teacher_first_layer.register_forward_hook(get_teacher_first_hook('teacher_first'))
        teacher_last_handle = teacher_last_layer.register_forward_hook(get_teacher_last_hook('teacher_last'))
        self.teacher_handles = [teacher_first_handle, teacher_last_handle]
        
        # 注册学生模型的hook
        student_first_handle = student_first_layer.register_forward_hook(get_student_first_hook('student_first'))
        student_last_handle = student_last_layer.register_forward_hook(get_student_last_hook('student_last'))
        self.student_handles = [student_first_handle, student_last_handle]
        
        print("已注册特征提取hook（第一层和最后一层）")
    
    def _remove_hooks(self):
        """移除所有hook"""
        for handle in self.teacher_handles:
            handle.remove()
        for handle in self.student_handles:
            handle.remove()
        self.teacher_handles = []
        self.student_handles = []
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算蒸馏损失（MSE 损失，因为时间序列预测是回归任务）
        
        Parameters
        ----------
        student_logits : torch.Tensor
            学生模型的预测值
        teacher_logits : torch.Tensor
            教师模型的预测值
        labels : torch.Tensor
            真实标签
            
        Returns
        -------
        torch.Tensor
            蒸馏损失
        """
        # 对于时间序列预测，使用 MSE 损失
        # 软标签损失：学生模型应该接近教师模型的预测
        soft_loss = F.mse_loss(student_logits, teacher_logits)
        
        # 硬标签损失：学生模型应该接近真实标签
        hard_loss = F.mse_loss(student_logits, labels)
        
        # 组合损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss
    
    def train_epoch(self, epoch: int):
        """训练一个 epoch"""
        self.student_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            context = batch["target"].to(self.device)  # [batch_size, context_length]
            future_target = batch["future_target"].to(self.device)  # [batch_size, horizon]
            
            batch_size = context.shape[0]
            
            # 准备输入格式（Chronos-2 格式）
            inputs = [{"target": context[i].cpu().numpy()} for i in range(batch_size)]
            
            # Chronos2Model.forward() 需要 context tensor，形状为 (batch_size, context_length)
            # 还需要 group_ids，形状为 (batch_size,)
            group_ids = torch.arange(batch_size, device=self.device)
            
            # 计算需要预测的 patch 数量
            output_patch_size = self.student_model.chronos_config.output_patch_size
            num_output_patches = (self.horizon + output_patch_size - 1) // output_patch_size
            
            # 清空特征缓存
            self.teacher_features = {}
            self.student_features = {}
            
            # 教师模型前向传播（不计算梯度，用于提取特征和预测）
            teacher_predictions = None
            teacher_logits = None
            with torch.no_grad():
                try:
                    # 先调用教师模型的forward来提取特征（hook会自动提取）
                    teacher_model = self.teacher_pipeline.model
                    teacher_outputs = teacher_model(
                        context=context,
                        group_ids=group_ids,
                        num_output_patches=num_output_patches
                    )
                    
                    # 从forward输出中提取预测值
                    teacher_prediction = teacher_outputs.quantile_preds
                    n_quantiles = teacher_prediction.shape[1]
                    median_idx = n_quantiles // 2
                    teacher_logits = teacher_prediction[:, median_idx, :]  # [batch_size, prediction_length]
                    
                    # 如果预测长度不匹配，截断或填充
                    if teacher_logits.shape[1] > self.horizon:
                        teacher_logits = teacher_logits[:, :self.horizon]
                    elif teacher_logits.shape[1] < self.horizon:
                        padding = torch.zeros(
                            batch_size,
                            self.horizon - teacher_logits.shape[1],
                            device=self.device
                        )
                        teacher_logits = torch.cat([teacher_logits, padding], dim=1)
                        
                except Exception as e:
                    print(f"警告: 教师模型前向传播失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 如果教师模型前向传播失败，跳过这个 batch
                    continue
            
            # 学生模型前向传播
            self.optimizer.zero_grad()
            
            try:
                # 调用学生模型的 forward 方法
                student_outputs = self.student_model(
                    context=context,
                    group_ids=group_ids,
                    num_output_patches=num_output_patches
                )
                
                # student_outputs 是 Chronos2Output，包含 quantile_preds 字段
                # quantile_preds 形状: (batch_size, n_quantiles, num_output_patches * output_patch_size)
                student_prediction = student_outputs.quantile_preds
                
                # 提取中位数预测
                n_quantiles = student_prediction.shape[1]
                median_idx = n_quantiles // 2
                student_logits = student_prediction[:, median_idx, :]  # [batch_size, prediction_length]
                
                # 如果预测长度不匹配，截断或填充
                if student_logits.shape[1] > self.horizon:
                    student_logits = student_logits[:, :self.horizon]
                elif student_logits.shape[1] < self.horizon:
                    # 填充到 horizon 长度
                    padding = torch.zeros(
                        batch_size,
                        self.horizon - student_logits.shape[1],
                        device=self.device
                    )
                    student_logits = torch.cat([student_logits, padding], dim=1)
                    
            except Exception as e:
                print(f"警告: 学生模型前向传播失败: {e}")
                import traceback
                traceback.print_exc()
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
            # 对于时间序列预测，我们使用 MSE 损失而不是分类损失
            mse_loss = F.mse_loss(student_logits, future_target)
            
            # 蒸馏损失：学生模型应该接近教师模型的预测
            distillation_loss = F.mse_loss(student_logits, teacher_logits)
            
            # 特征蒸馏损失（第一层和最后一层）
            feature_distill_loss = torch.tensor(0.0, device=self.device)
            feature_loss_first = torch.tensor(0.0, device=self.device)
            feature_loss_last = torch.tensor(0.0, device=self.device)
            
            if 'first' in self.teacher_features and 'first' in self.student_features:
                teacher_feat_first = self.teacher_features['first']
                student_feat_first = self.student_features['first']
                
                # 使用regressor将教师特征转换为学生维度
                teacher_feat_first_aligned = self.feature_regressor_first(teacher_feat_first)
                
                # 计算第一层特征损失（MSE）
                feature_loss_first = F.mse_loss(student_feat_first, teacher_feat_first_aligned)
                feature_distill_loss += feature_loss_first
            
            if 'last' in self.teacher_features and 'last' in self.student_features:
                teacher_feat_last = self.teacher_features['last']
                student_feat_last = self.student_features['last']
                
                # 使用regressor将教师特征转换为学生维度
                teacher_feat_last_aligned = self.feature_regressor_last(teacher_feat_last)
                
                # 计算最后一层特征损失（MSE）
                feature_loss_last = F.mse_loss(student_feat_last, teacher_feat_last_aligned)
                feature_distill_loss += feature_loss_last
            
            # DTW损失：计算学生预测与真实标签之间的DTW距离
            dtw_loss = torch.tensor(0.0, device=self.device)
            if self.dtw_weight > 0:  # 只有当权重>0时才计算DTW
                try:
                    # 确保序列长度匹配
                    min_len = min(student_logits.shape[1], future_target.shape[1])
                    if min_len > 0:
                        if batch_idx == 0 and epoch == 0:
                            print(f"  计算DTW损失 (序列长度: {min_len}, batch_size: {batch_size})...")
                        dtw_loss = self.dtw_loss_fn(
                            student_logits[:, :min_len],
                            future_target[:, :min_len]
                        )
                        # 确保DTW损失非负
                        dtw_loss = torch.clamp(dtw_loss, min=0.0)
                        if batch_idx == 0 and epoch == 0:
                            print(f"  DTW损失计算完成: {dtw_loss.item():.4f}")
                except Exception as e:
                    print(f"警告: DTW损失计算失败: {e}")
                    import traceback
                    traceback.print_exc()
                    dtw_loss = torch.tensor(0.0, device=self.device)
            
            # 检查损失是否为 NaN 或 Inf
            if torch.isnan(mse_loss) or torch.isinf(mse_loss) or \
               torch.isnan(distillation_loss) or torch.isinf(distillation_loss) or \
               torch.isnan(feature_distill_loss) or torch.isinf(feature_distill_loss) or \
               torch.isnan(dtw_loss) or torch.isinf(dtw_loss):
                print(f"警告: Batch {batch_idx + 1} 损失包含 NaN/Inf，跳过")
                continue
            
            # 组合损失：预测蒸馏损失 + 特征蒸馏损失 + 真实标签损失 + DTW损失
            # alpha: 预测蒸馏损失权重, beta: 特征蒸馏损失权重, dtw_weight: DTW损失权重
            beta = 0.3  # 特征蒸馏损失权重
            loss = (self.alpha * distillation_loss + 
                   (1 - self.alpha) * mse_loss + 
                   beta * feature_distill_loss +
                   self.dtw_weight * dtw_loss)
            
            # 检查损失值是否异常（负数或过大）
            loss_value = loss.item()
            if loss_value < 0:
                print(f"警告: Batch {batch_idx + 1} 损失为负数 ({loss_value:.4f})，跳过")
                print(f"  MSE: {mse_loss.item():.4f}, Distill: {distillation_loss.item():.4f}, "
                      f"FeatDistill: {feature_distill_loss.item():.4f}, DTW: {dtw_loss.item():.4f}")
                self.optimizer.zero_grad()
                continue
            if loss_value > 1e6:
                print(f"警告: Batch {batch_idx + 1} 损失过大 ({loss_value:.2f})，跳过")
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            grad_norm = torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.max_grad_norm)
            
            # 记录梯度范数（用于调试）
            if (batch_idx + 1) % 100 == 0:
                print(f"  梯度范数: {grad_norm.item():.4f}")
            
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
                continue
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                feat_loss_str = f", FeatDistill: {feature_distill_loss.item():.4f}" if feature_distill_loss.item() > 0 else ""
                feat_first_str = f", FeatFirst: {feature_loss_first.item():.4f}" if feature_loss_first.item() > 0 else ""
                feat_last_str = f", FeatLast: {feature_loss_last.item():.4f}" if feature_loss_last.item() > 0 else ""
                dtw_str = f", DTW: {dtw_loss.item():.4f}"
                print(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, MSE: {mse_loss.item():.4f}, Distill: {distillation_loss.item():.4f}"
                      f"{feat_loss_str}{feat_first_str}{feat_last_str}{dtw_str}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def evaluate(self):
        """评估模型，返回MSE损失和DTW距离"""
        if self.eval_loader is None:
            return None
        
        self.student_model.eval()
        total_mse_loss = 0.0
        total_dtw_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                context = batch["target"].to(self.device)
                future_target = batch["future_target"].to(self.device)
                
                batch_size = context.shape[0]
                group_ids = torch.arange(batch_size, device=self.device)
                
                # 计算需要预测的 patch 数量
                output_patch_size = self.student_model.chronos_config.output_patch_size
                num_output_patches = (self.horizon + output_patch_size - 1) // output_patch_size
                
                try:
                    student_outputs = self.student_model(
                        context=context,
                        group_ids=group_ids,
                        num_output_patches=num_output_patches
                    )
                    
                    # 提取中位数预测
                    student_prediction = student_outputs.quantile_preds
                    n_quantiles = student_prediction.shape[1]
                    median_idx = n_quantiles // 2
                    student_logits = student_prediction[:, median_idx, :]
                    
                    # 调整长度
                    if student_logits.shape[1] > self.horizon:
                        student_logits = student_logits[:, :self.horizon]
                    elif student_logits.shape[1] < self.horizon:
                        padding = torch.zeros(
                            batch_size,
                            self.horizon - student_logits.shape[1],
                            device=self.device
                        )
                        student_logits = torch.cat([student_logits, padding], dim=1)
                    
                    # 计算 MSE 损失和 DTW 距离
                    min_len = min(student_logits.shape[1], future_target.shape[1])
                    mse_loss = F.mse_loss(student_logits[:, :min_len], future_target[:, :min_len])
                    
                    # 计算DTW距离
                    try:
                        dtw_loss = self.dtw_loss_fn(
                            student_logits[:, :min_len],
                            future_target[:, :min_len]
                        )
                    except Exception as e:
                        dtw_loss = torch.tensor(0.0, device=self.device)
                    
                    total_mse_loss += mse_loss.item()
                    total_dtw_loss += dtw_loss.item()
                    num_batches += 1
                except Exception as e:
                    print(f"评估时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        avg_mse_loss = total_mse_loss / num_batches if num_batches > 0 else 0.0
        avg_dtw_loss = total_dtw_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'mse_loss': avg_mse_loss,
            'dtw_loss': avg_dtw_loss
        }
    
    def train(self, resume_from_epoch: Optional[int] = None, resume_checkpoint_path: Optional[str] = None):
        """
        执行训练
        
        Parameters
        ----------
        resume_from_epoch : Optional[int]
            从哪个epoch继续训练（如果指定了resume_checkpoint_path，此参数将被忽略）
        resume_checkpoint_path : Optional[str]
            从指定checkpoint路径恢复训练
        """
        print("=" * 50)
        print("开始 Chronos-2 蒸馏训练")
        print("=" * 50)
        
        start_epoch = 0
        best_eval_loss = float('inf')
        
        # 如果指定了checkpoint路径，从checkpoint恢复
        if resume_checkpoint_path:
            if not os.path.exists(resume_checkpoint_path):
                raise FileNotFoundError(f"Checkpoint路径不存在: {resume_checkpoint_path}")
            start_epoch, best_eval_loss = self.load_checkpoint(resume_checkpoint_path)
            print(f"从epoch {start_epoch + 1}继续训练，最佳验证损失: {best_eval_loss:.4f}")
        # 如果只指定了epoch编号，尝试从默认路径加载
        elif resume_from_epoch is not None:
            default_checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{resume_from_epoch}")
            if os.path.exists(default_checkpoint_path):
                start_epoch, best_eval_loss = self.load_checkpoint(default_checkpoint_path)
                print(f"从epoch {start_epoch + 1}继续训练，最佳验证损失: {best_eval_loss:.4f}")
            else:
                # 如果没有找到checkpoint，尝试从best_model加载
                best_model_path = os.path.join(self.output_dir, f"best_model_epoch_{resume_from_epoch}")
                if os.path.exists(best_model_path):
                    print(f"警告: 未找到checkpoint，尝试从best_model加载模型（不会恢复optimizer状态）")
                    self.student_model = Chronos2Model.from_pretrained(best_model_path)
                    self.student_model = move_model_to_device(self.student_model, self.device)
                    start_epoch = resume_from_epoch - 1
                    print(f"从epoch {start_epoch + 1}继续训练（仅恢复模型权重）")
                else:
                    raise FileNotFoundError(
                        f"未找到epoch {resume_from_epoch}的checkpoint或模型。"
                        f"请检查 {default_checkpoint_path} 或 {best_model_path} 是否存在。"
                    )
        
        for epoch in range(start_epoch, self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"训练损失: {train_loss:.4f}")
            
            # 评估
            if self.eval_loader:
                eval_metrics = self.evaluate()
                if eval_metrics is not None:
                    eval_mse_loss = eval_metrics['mse_loss']
                    eval_dtw_loss = eval_metrics['dtw_loss']
                    print(f"验证MSE损失: {eval_mse_loss:.4f}, DTW损失: {eval_dtw_loss:.4f}")
                    
                    # 使用MSE损失作为主要指标来选择最佳模型
                    if eval_mse_loss < best_eval_loss:
                        best_eval_loss = eval_mse_loss
                        self.save_model(f"best_model_epoch_{epoch + 1}")
                        print(f"保存最佳模型 (验证MSE损失: {eval_mse_loss:.4f}, DTW损失: {eval_dtw_loss:.4f})")
            
            # 每个epoch结束后保存checkpoint
            self.save_checkpoint(epoch, best_eval_loss, checkpoint_name="checkpoint")
        
        # 保存最终模型
        self.save_model("final_model")
        print("\n训练完成！")
    
    def save_model(self, model_name: str):
        """保存模型"""
        save_path = os.path.join(self.output_dir, model_name)
        os.makedirs(save_path, exist_ok=True)
        self.student_model.save_pretrained(save_path)
        print(f"模型已保存到: {save_path}")
    
    def save_checkpoint(self, epoch: int, best_eval_loss: float, checkpoint_name: str = "checkpoint"):
        """
        保存完整的checkpoint（包括模型、optimizer状态、epoch等信息）
        
        Parameters
        ----------
        epoch : int
            当前epoch编号
        best_eval_loss : float
            当前最佳验证损失
        checkpoint_name : str
            checkpoint名称
        """
        checkpoint_path = os.path.join(self.output_dir, f"{checkpoint_name}_epoch_{epoch + 1}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(checkpoint_path, "model")
        os.makedirs(model_path, exist_ok=True)
        self.student_model.save_pretrained(model_path)
        
        # 保存特征回归器
        regressor_path = os.path.join(checkpoint_path, "regressors.pt")
        torch.save({
            'feature_regressor_first': self.feature_regressor_first.state_dict(),
            'feature_regressor_last': self.feature_regressor_last.state_dict(),
        }, regressor_path)
        
        # 保存训练状态
        checkpoint_file = os.path.join(checkpoint_path, "training_state.pt")
        torch.save({
            'epoch': epoch,
            'best_eval_loss': best_eval_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'alpha': self.alpha,
            'temperature': self.temperature,
        }, checkpoint_file)
        
        print(f"Checkpoint已保存到: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        从checkpoint加载训练状态
        
        Parameters
        ----------
        checkpoint_path : str
            checkpoint路径（应该是包含training_state.pt和model目录的路径）
            
        Returns
        -------
        tuple
            (epoch, best_eval_loss) 恢复的epoch编号和最佳验证损失
        """
        model_path = os.path.join(checkpoint_path, "model")
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        regressor_path = os.path.join(checkpoint_path, "regressors.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型目录不存在: {model_path}")
        if not os.path.exists(training_state_path):
            raise FileNotFoundError(f"训练状态文件不存在: {training_state_path}")
        
        print(f"从checkpoint加载: {checkpoint_path}")
        
        # 加载模型
        print("加载学生模型...")
        self.student_model = Chronos2Model.from_pretrained(model_path)
        self.student_model = move_model_to_device(self.student_model, self.device)
        
        # 加载特征回归器
        if os.path.exists(regressor_path):
            print("加载特征回归器...")
            regressor_state = torch.load(regressor_path, map_location=self.device)
            self.feature_regressor_first.load_state_dict(regressor_state['feature_regressor_first'])
            self.feature_regressor_last.load_state_dict(regressor_state['feature_regressor_last'])
        
        # 加载训练状态
        print("加载训练状态...")
        checkpoint = torch.load(training_state_path, map_location=self.device)
        
        epoch = checkpoint.get('epoch', 0)
        best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        
        # 重新创建optimizer（因为模型参数可能已改变）
        optimizer_params = list(self.student_model.parameters()) + \
                          list(self.feature_regressor_first.parameters()) + \
                          list(self.feature_regressor_last.parameters())
        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=checkpoint.get('learning_rate', self.learning_rate),
            weight_decay=1e-4,
            eps=1e-8
        )
        
        # 加载optimizer状态
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer状态已加载")
            except Exception as e:
                print(f"警告: 无法加载optimizer状态，将使用新的optimizer状态: {e}")
        
        print(f"成功加载checkpoint: epoch={epoch + 1}, best_eval_loss={best_eval_loss:.4f}")
        return epoch, best_eval_loss


def create_student_model(
    teacher_pipeline: Chronos2Pipeline,
    student_config_overrides: Optional[Dict[str, Any]] = None
) -> Chronos2Model:
    """
    创建学生模型（更小的架构）
    """
    teacher_config = teacher_pipeline.model.config
    
    # 创建学生模型配置（复制教师配置）
    student_config = teacher_config.__class__.from_dict(teacher_config.to_dict())
    
    # 应用配置覆盖
    if student_config_overrides:
        for key, value in student_config_overrides.items():
            if hasattr(student_config, key):
                setattr(student_config, key, value)
            elif hasattr(student_config, 'chronos_config'):
                chronos_config_dict = student_config.chronos_config.to_dict() if hasattr(student_config.chronos_config, 'to_dict') else student_config.chronos_config
                if key in chronos_config_dict:
                    chronos_config_dict[key] = value
                    student_config.chronos_config = Chronos2ForecastingConfig(**chronos_config_dict)
    
    # 创建学生模型
    student_model = Chronos2Model(student_config)
    
    return student_model


def main():
    """主函数"""
    
    # ========== 配置参数 ==========
    teacher_model_id = "amazon/chronos-2"  # 或 "amazon/chronos-2"
    output_dir = "./chronos-2-distilled"
    
    # 数据配置（参考 LightGTS 的处理方式）
    # data_dir = "data/datasets/Eval_Data/traffic" # 先用小数据集测试一下能否跑通
    data_dir = "data/datasets/Pretrain_Data"  # 数据目录（训练集和验证集都从此目录划分）
    
    # 方式1：指定数据集名称列表（推荐，更可控）
    # 可以从目录中选择特定的数据集，例如：
    # dataset_names = ['ETTh2', 'kdd_cup_2018_dataset_without_missing_values(0)']
    dataset_names = None  # None 表示加载目录下所有 CSV 文件
    
    # 方式2：直接指定文件路径
    data_paths = []  # 也可以指定具体的文件路径列表
    
    # 数据划分比例（8:2 划分：训练集80%，验证集20%）
    train_split = 0.8  # 训练集比例
    test_split = 0.0   # 测试集比例（设为0表示不保留测试集，全部用于训练和验证）
    
    context_length = 720
    horizon = 96
    stride = 96
    target_col = None  # None 表示自动检测（通常是最后一列或第一个数值列）
    
    # 学生模型配置
    student_config_overrides = {
        "num_layers": 6,      # 减少层数
        "d_model": 384,       # 减少隐藏层维度
        "d_ff": 1536,         # 减少前馈网络维度
        "num_heads": 6,       # 减少注意力头数
    }
    
    # 训练配置
    learning_rate = 1e-5  # 降低学习率，提高稳定性
    batch_size = 256
    num_epochs = 20
    temperature = 2.0
    alpha = 0.5  # 蒸馏损失权重
    max_grad_norm = 1.0  # 梯度裁剪阈值
    
    # DTW损失配置
    dtw_gamma = 1.0  # DTW损失的温度参数（gamma越小越接近硬DTW，gamma越大越平滑）
    dtw_weight = 0.2  # DTW损失的权重（建议范围：0.1-0.3）
    
    # ========== 加载教师模型 ==========
    print("=" * 50)
    print("加载教师模型 (Chronos-2)...")
    print("=" * 50)
    
    # 确定设备（使用单个设备，避免多设备分布导致的问题）
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 不使用 device_map，让模型加载到CPU，然后在训练器初始化时移动到指定设备
    # 这样可以避免多设备分布导致的设备不匹配问题
    
    teacher_pipeline = Chronos2Pipeline.from_pretrained(
        teacher_model_id,
        device_map=None  # 不使用 device_map，加载到CPU
    )
    # 确保模型完全在CPU上（避免accelerate自动分配设备）
    # 注意：如果模型使用了accelerate hooks，这里会触发警告，但应该仍然可以工作
    try:
        teacher_pipeline.model = teacher_pipeline.model.to("cpu")
    except RuntimeError as e:
        if "accelerate hooks" in str(e).lower():
            print("警告：模型使用了accelerate hooks，跳过CPU移动。将在初始化时处理设备问题。")
        else:
            raise
    print(f"教师模型加载完成: {teacher_model_id}")
    teacher_params = sum(p.numel() for p in teacher_pipeline.model.parameters())
    print(f"教师模型参数量: {teacher_params:,}")
    
    # ========== 创建学生模型 ==========
    print("\n" + "=" * 50)
    print("创建学生模型...")
    print("=" * 50)
    student_model = create_student_model(
        teacher_pipeline,
        student_config_overrides
    )
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"学生模型参数量: {student_params:,}")
    print(f"压缩比: {student_params / teacher_params:.2%}")
    
    # ========== 准备数据 ==========
    print("\n" + "=" * 50)
    print("准备训练数据...")
    print("=" * 50)
    
    # 训练数据集：从数据目录加载，使用8:2划分中的训练集部分
    train_dataset = Chronos2DistillationDataset(
        data_paths=data_paths,
        data_dir=data_dir,
        dataset_names=dataset_names,  # 如果指定，只加载这些数据集
        context_length=context_length,
        horizon=horizon,
        stride=stride,
        target_col=target_col,
        scale=True,  # 数据标准化（参考 LightGTS）
        split='train',  # 训练集
        train_split=train_split,
        test_split=test_split
    )
    
    # 验证数据集：从同一数据目录加载，使用8:2划分中的验证集部分
    print("\n" + "=" * 50)
    print("准备验证数据...")
    print("=" * 50)
    eval_dataset = Chronos2DistillationDataset(
        data_paths=data_paths,
        data_dir=data_dir,
        dataset_names=dataset_names,
        context_length=context_length,
        horizon=horizon,
        stride=horizon,  # 验证时使用更大的步长，避免重叠
        target_col=target_col,
        scale=True,  # 数据标准化（使用与训练集相同的scaler）
        split='val',  # 验证集
        train_split=train_split,
        test_split=test_split
    )
    
    print(f"\n训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset)}")
    
    # ========== 创建训练器并训练 ==========
    trainer = Chronos2DistillationTrainer(
        teacher_pipeline=teacher_pipeline,
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
        device=device,  # 传递设备参数
        output_dir=output_dir,
        max_grad_norm=max_grad_norm,
        dtw_gamma=dtw_gamma,  # DTW损失温度参数
        dtw_weight=dtw_weight  # DTW损失权重
    )
    
    # ========== 继续训练选项 ==========
    # 如果需要从某个epoch继续训练，设置resume_from_epoch参数
    # 例如：resume_from_epoch = 9  # 从第9个epoch继续（会从checkpoint_epoch_9或best_model_epoch_9加载）
    # 或者直接指定checkpoint路径：
    # resume_checkpoint_path = "./chronos-2-distilled/checkpoint_epoch_9"
    resume_from_epoch = None  # 设置为None表示从头开始训练，或者设置为epoch编号（从1开始，例如9表示从第9个epoch继续）
    resume_checkpoint_path = None  # 直接指定checkpoint路径（优先级高于resume_from_epoch）
    
    trainer.train(resume_from_epoch=resume_from_epoch, resume_checkpoint_path=resume_checkpoint_path)
    
    print("\n" + "=" * 50)
    print("蒸馏完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()

