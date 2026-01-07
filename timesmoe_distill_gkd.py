"""
TimesMoE 蒸馏脚本 - 使用 TRL GKD 的适配方案

基于 chronos2_distill_gkd.py 的结构，适配 TimesMoE 模型进行知识蒸馏。
本脚本提供了针对 TimesMoE 的蒸馏实现。
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
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 尝试导入 TimesMoE（如果可用）
try:
    from timesmoe import TimesMoEModel
    TIMESMOE_AVAILABLE = True
except ImportError:
    TIMESMOE_AVAILABLE = False
    print("警告: timesmoe 模块未安装，请先安装: pip install timesmoe")


class TimesMoEDistillationDataset(Dataset):
    """TimesMoE 蒸馏数据集"""
    
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
        time_col_name: str = 'date'
    ):
        self.context_length = context_length
        self.horizon = horizon
        self.scale = scale
        self.split = split
        self.time_col_name = time_col_name
        self.samples = []
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
            data_paths = [str(f) for f in csv_files]
        
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
                
                # 使用滑动窗口创建样本
                max_start_idx = len(ts_values) - context_length - horizon
                
                file_samples = 0
                for start_idx in range(0, max_start_idx + 1, stride):
                    context = ts_values[start_idx:start_idx + context_length]
                    target = ts_values[start_idx + context_length:start_idx + context_length + horizon]
                    
                    if np.isnan(context).any() or np.isnan(target).any():
                        continue
                    
                    self.samples.append({
                        "target": torch.tensor(context, dtype=torch.float32),
                        "future_target": torch.tensor(target, dtype=torch.float32)
                    })
                    file_samples += 1
                
                if file_samples > 0:
                    print(f"  {os.path.basename(data_path)} ({split}): 添加了 {file_samples} 个样本")
                    
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


class SimpleTimesMoEStudent(nn.Module):
    """
    简单的 TimesMoE 学生模型
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


class TimesMoEDistillationTrainer:
    """TimesMoE 知识蒸馏训练器"""
    
    def __init__(
        self,
        teacher_model: Any,  # TimesMoE 模型
        student_model: nn.Module,
        train_dataset: TimesMoEDistillationDataset,
        eval_dataset: Optional[TimesMoEDistillationDataset] = None,
        context_length: int = 96,
        horizon: int = 24,
        learning_rate: float = 5e-5,
        batch_size: int = 32,
        num_epochs: int = 10,
        temperature: float = 2.0,
        alpha: float = 0.5,  # 蒸馏损失权重
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./timesmoe-distilled",
        max_grad_norm: float = 1.0  # 梯度裁剪阈值
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
        使用教师模型（TimesMoE）进行预测
        
        注意：此方法需要根据 TimesMoE 的实际 API 进行调整
        TimesMoE 的预测接口可能与 Chronos2 不同
        
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
        teacher_predictions = []
        
        # 将 tensor 转换为 numpy（如果需要）
        context_np = context_batch.cpu().numpy()
        
        for i in range(batch_size):
            context_series = context_np[i]  # (context_length,)
            
            # 根据 TimesMoE 的实际 API 调整预测方法
            # 以下是可能的 API 示例，需要根据实际情况修改：
            try:
                # 方式1：如果 TimesMoE 有 predict 方法，接受 numpy array
                if hasattr(self.teacher_model, 'forecast') or hasattr(self.teacher_model, 'predict'):
                    # TimesMoE 可能接受 2D 输入 (1, context_length) 或 1D (context_length,)
                    if context_series.ndim == 1:
                        context_series = context_series.reshape(1, -1)  # (1, context_length)
                    
                    # 尝试 forecast 方法
                    if hasattr(self.teacher_model, 'forecast'):
                        pred = self.teacher_model.forecast(
                            context_series,
                            forecast_length=self.horizon
                        )
                    # 尝试 predict 方法
                    elif hasattr(self.teacher_model, 'predict'):
                        pred = self.teacher_model.predict(
                            context_series,
                            prediction_length=self.horizon
                        )
                    else:
                        # 直接调用模型（如果是 PyTorch 模型）
                        context_tensor = torch.tensor(context_series, dtype=torch.float32).to(self.device)
                        if context_tensor.ndim == 1:
                            context_tensor = context_tensor.unsqueeze(0)  # (1, context_length)
                        pred = self.teacher_model(context_tensor)
                    
                    # 处理预测结果格式
                    if isinstance(pred, (list, tuple)):
                        pred = pred[0]
                    if torch.is_tensor(pred):
                        pred = pred.detach().cpu().numpy()
                    if isinstance(pred, np.ndarray):
                        if pred.ndim > 1:
                            pred = pred.flatten()
                        pred = pred[:self.horizon]  # 确保长度正确
                    else:
                        raise ValueError(f"不支持的预测结果类型: {type(pred)}")
                    
                    teacher_predictions.append(pred)
                else:
                    # 如果模型是 PyTorch 模型，直接调用 forward
                    if hasattr(self.teacher_model, '__call__'):
                        context_tensor = torch.tensor(context_series, dtype=torch.float32).to(self.device)
                        if context_tensor.ndim == 1:
                            context_tensor = context_tensor.unsqueeze(0)  # (1, context_length)
                        
                        with torch.no_grad():
                            pred = self.teacher_model(context_tensor)
                        
                        if torch.is_tensor(pred):
                            pred = pred.detach().cpu().numpy()
                        if pred.ndim > 1:
                            pred = pred.flatten()
                        pred = pred[:self.horizon]
                        teacher_predictions.append(pred)
                    else:
                        raise NotImplementedError(
                            "TimesMoE 模型需要实现 predict、forecast 方法或 __call__ 方法。"
                            "请根据实际的 TimesMoE API 修改 _teacher_predict 方法。"
                        )
            except Exception as e:
                print(f"警告: 教师模型预测单个样本失败: {e}")
                import traceback
                traceback.print_exc()
                # 使用零填充作为后备
                teacher_predictions.append(np.zeros(self.horizon, dtype=np.float32))
        
        # 堆叠为 tensor: [batch_size, horizon]
        teacher_logits = torch.tensor(np.stack(teacher_predictions), dtype=torch.float32).to(self.device)
        return teacher_logits
    
    def train_epoch(self, epoch: int):
        """训练一个 epoch"""
        self.student_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            context = batch["target"].to(self.device)  # [batch_size, context_length]
            future_target = batch["future_target"].to(self.device)  # [batch_size, horizon]
            
            batch_size = context.shape[0]
            
            # 教师模型预测（不计算梯度）
            with torch.no_grad():
                try:
                    teacher_logits = self._teacher_predict(context)
                except Exception as e:
                    print(f"警告: 教师模型预测失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 学生模型前向传播
            self.optimizer.zero_grad()
            
            try:
                # 调用学生模型的 forward 方法
                # SimpleTimesMoEStudent 直接接受 context 并输出预测
                student_logits = self.student_model(context)  # [batch_size, horizon]
                    
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
            mse_loss = F.mse_loss(student_logits, future_target)
            distillation_loss = F.mse_loss(student_logits, teacher_logits)
            
            # 检查损失是否为 NaN 或 Inf
            if torch.isnan(mse_loss) or torch.isinf(mse_loss) or torch.isnan(distillation_loss) or torch.isinf(distillation_loss):
                print(f"警告: Batch {batch_idx + 1} 损失包含 NaN/Inf，跳过")
                continue
            
            # 组合损失
            loss = self.alpha * distillation_loss + (1 - self.alpha) * mse_loss
            
            # 限制损失值，防止数值爆炸
            if loss.item() > 1e6:
                print(f"警告: Batch {batch_idx + 1} 损失过大 ({loss.item():.2f})，跳过")
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.max_grad_norm)
            
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
                print(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, MSE: {mse_loss.item():.4f}, Distill: {distillation_loss.item():.4f}")
        
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
        print("开始 TimesMoE 蒸馏训练")
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
) -> SimpleTimesMoEStudent:
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
    SimpleTimesMoEStudent
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
    
    student_model = SimpleTimesMoEStudent(**config)
    return student_model


def main():
    """主函数"""
    
    if not TIMESMOE_AVAILABLE:
        print("错误: timesmoe 模块未安装。请先安装: pip install timesmoe")
        return
    
    # ========== 配置参数 ==========
    teacher_model_id = "google/timesmoe-1.0-200m"  # TimesMoE 模型 ID，需要根据实际情况调整
    output_dir = "./timesmoe-distilled"
    
    # 数据配置
    train_data_dir = "datasets/monash_csv_downsmp"
    train_dataset_names = None  # None 表示加载目录下所有 CSV 文件
    train_data_paths = []
    
    # 验证数据
    eval_data_paths = ["datasets/monash_csv_downsmp"]
    eval_dataset_names = None
    
    context_length = 96
    horizon = 24
    stride = 1
    target_col = None
    
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
    batch_size = 32
    num_epochs = 10
    temperature = 2.0
    alpha = 0.5
    max_grad_norm = 1.0
    
    # ========== 加载教师模型 ==========
    print("=" * 50)
    print("加载教师模型 (TimesMoE)...")
    print("=" * 50)
    try:
        teacher_model = TimesMoEModel.from_pretrained(teacher_model_id)
        print(f"教师模型加载完成: {teacher_model_id}")
        # 计算参数量（如果可能）
        if hasattr(teacher_model, 'parameters'):
            teacher_params = sum(p.numel() for p in teacher_model.parameters())
            print(f"教师模型参数量: {teacher_params:,}")
    except Exception as e:
        print(f"加载教师模型失败: {e}")
        print("请检查模型 ID 是否正确，以及 timesmoe 库是否已正确安装。")
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
    
    train_dataset = TimesMoEDistillationDataset(
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
        test_split=0.1
    )
    
    print("\n" + "=" * 50)
    print("准备验证数据...")
    print("=" * 50)
    eval_dataset = TimesMoEDistillationDataset(
        data_paths=eval_data_paths,
        dataset_names=eval_dataset_names,
        context_length=context_length,
        horizon=horizon,
        stride=horizon,
        target_col=target_col,
        scale=True,
        split='val',
        train_split=0.8,
        test_split=0.1
    )
    
    print(f"\n训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset)}")
    
    # ========== 创建训练器并训练 ==========
    trainer = TimesMoEDistillationTrainer(
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
        max_grad_norm=max_grad_norm
    )
    
    trainer.train()
    
    print("\n" + "=" * 50)
    print("蒸馏完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()

