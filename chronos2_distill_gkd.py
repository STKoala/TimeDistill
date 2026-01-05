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
        max_grad_norm: float = 1.0  # 梯度裁剪阈值
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
        
        # 将模型移到设备
        self.teacher_pipeline.model = self.teacher_pipeline.model.to(device)
        self.student_model = self.student_model.to(device)
        
        # 设置教师模型为评估模式并冻结所有参数
        self.teacher_pipeline.model.eval()
        for param in self.teacher_pipeline.model.parameters():
            param.requires_grad = False
        print("教师模型已设置为评估模式并冻结所有参数")
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
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
            
            # 教师模型预测（不计算梯度）
            teacher_predictions = None
            with torch.no_grad():
                try:
                    teacher_outputs = self.teacher_pipeline.predict(
                        inputs,
                        prediction_length=self.horizon
                    )
                    # teacher_outputs 是列表，每个元素是 (n_variates, n_quantiles, prediction_length)
                    # 提取中位数预测 (quantile 0.5，通常是中间位置)
                    teacher_predictions = []
                    for output in teacher_outputs:
                        if output.ndim == 3:  # (n_variates, n_quantiles, prediction_length)
                            # 取第一个 variate，中位数 quantile (通常是中间位置)
                            n_quantiles = output.shape[1]
                            median_idx = n_quantiles // 2
                            teacher_predictions.append(output[0, median_idx, :])  # (prediction_length,)
                        else:
                            teacher_predictions.append(output.squeeze())
                    
                    # 堆叠为 tensor: [batch_size, prediction_length]
                    teacher_logits = torch.stack(teacher_predictions).to(self.device)
                except Exception as e:
                    print(f"警告: 教师模型预测失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 如果教师模型预测失败，跳过这个 batch
                    continue
            
            # 学生模型前向传播
            self.optimizer.zero_grad()
            
            # Chronos2Model.forward() 需要 context tensor，形状为 (batch_size, context_length)
            # 还需要 group_ids，形状为 (batch_size,)
            group_ids = torch.arange(batch_size, device=self.device)
            
            # 计算需要预测的 patch 数量
            output_patch_size = self.student_model.chronos_config.output_patch_size
            num_output_patches = (self.horizon + output_patch_size - 1) // output_patch_size
            
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
                    
                    # 计算 MSE 损失
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
        print("开始 Chronos-2 蒸馏训练")
        print("=" * 50)
        
        best_eval_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"训练损失: {train_loss:.4f}")
            
            # 评估
            if self.eval_loader:
                eval_loss = self.evaluate()
                if eval_loss is not None:
                    print(f"验证损失: {eval_loss:.4f}")
                    
                    # 保存最佳模型
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self.save_model(f"best_model_epoch_{epoch + 1}")
                        print(f"保存最佳模型 (验证损失: {eval_loss:.4f})")
        
        # 保存最终模型
        self.save_model("final_model")
        print("\n训练完成！")
    
    def save_model(self, model_name: str):
        """保存模型"""
        save_path = os.path.join(self.output_dir, model_name)
        os.makedirs(save_path, exist_ok=True)
        self.student_model.save_pretrained(save_path)
        print(f"模型已保存到: {save_path}")


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
    train_data_dir = "datasets/monash_csv_downsmp"  # 训练数据目录
    
    # 方式1：指定数据集名称列表（推荐，更可控）
    # 可以从目录中选择特定的数据集，例如：
    # train_dataset_names = ['ETTh2', 'kdd_cup_2018_dataset_without_missing_values(0)']
    train_dataset_names = None  # None 表示加载目录下所有 CSV 文件
    
    # 方式2：直接指定文件路径
    train_data_paths = []  # 也可以指定具体的文件路径列表
    
    # 验证数据：使用 ETTh1 作为验证集
    eval_data_paths = [
        "datasets/ETTh1.csv",
    ]
    eval_dataset_names = None  # 验证集也可以使用数据集名称列表
    
    context_length = 720
    horizon = 96
    stride = 1
    target_col = None  # None 表示自动检测（通常是最后一列或第一个数值列）
    
    # 学生模型配置
    student_config_overrides = {
        "num_layers": 3,      # 减少层数
        "d_model": 384,       # 减少隐藏层维度
        "d_ff": 1536,         # 减少前馈网络维度
        "num_heads": 6,       # 减少注意力头数
    }
    
    # 训练配置
    learning_rate = 1e-5  # 降低学习率，提高稳定性
    batch_size = 256
    num_epochs = 10
    temperature = 2.0
    alpha = 0.5  # 蒸馏损失权重
    max_grad_norm = 1.0  # 梯度裁剪阈值
    
    # ========== 加载教师模型 ==========
    print("=" * 50)
    print("加载教师模型 (Chronos-2)...")
    print("=" * 50)
    teacher_pipeline = Chronos2Pipeline.from_pretrained(
        teacher_model_id,
        device_map="auto"
    )
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
    
    # 训练数据集：从 monash_csv_downsmp 目录加载（参考 LightGTS）
    train_dataset = Chronos2DistillationDataset(
        data_paths=train_data_paths,
        data_dir=train_data_dir,
        dataset_names=train_dataset_names,  # 如果指定，只加载这些数据集
        context_length=context_length,
        horizon=horizon,
        stride=stride,
        target_col=target_col,
        scale=True,  # 数据标准化（参考 LightGTS）
        split='train',  # 训练集
        train_split=0.8,
        test_split=0.1
    )
    
    # 验证数据集：使用 ETTh1 数据
    print("\n" + "=" * 50)
    print("准备验证数据...")
    print("=" * 50)
    eval_dataset = Chronos2DistillationDataset(
        data_paths=eval_data_paths,
        dataset_names=eval_dataset_names,
        context_length=context_length,
        horizon=horizon,
        stride=horizon,  # 验证时使用更大的步长，避免重叠
        target_col=target_col,
        scale=True,  # 数据标准化
        split='val',  # 验证集
        train_split=0.8,
        test_split=0.1
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
        output_dir=output_dir,
        max_grad_norm=max_grad_norm
    )
    
    trainer.train()
    
    print("\n" + "=" * 50)
    print("蒸馏完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()

