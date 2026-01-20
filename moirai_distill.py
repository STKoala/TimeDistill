"""
MOIRAI 知识蒸馏脚本

参考 uni2ts 中的 MOIRAI 源码，实现知识蒸馏：
- 单变量预测
- 720预测96（context_length=720, prediction_length=96）
- 使用 TimeDistill/data/datasets/Pretrain_Data 数据集
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from pathlib import Path

# 添加 uni2ts 路径
script_dir = Path(__file__).parent
uni2ts_paths = [
    script_dir.parent / "uni2ts" / "src",  # 从源码导入
    script_dir.parent / "uni2ts",  # 直接导入
]

for uni2ts_path in uni2ts_paths:
    if uni2ts_path.exists():
        sys.path.insert(0, str(uni2ts_path))
        print(f"添加 uni2ts 路径: {uni2ts_path}")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import argparse
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

# 导入 MOIRAI 相关模块
MOIRAI_AVAILABLE = False
MoiraiForecast = None
MoiraiModule = None
NormalOutput = None

# 尝试多种导入方式
import_errors = []

# 首先检查关键依赖
missing_deps = []
try:
    import lightning
except ImportError:
    missing_deps.append("lightning>=2.0")
try:
    import gluonts
except ImportError:
    missing_deps.append("gluonts~=0.14.3")
try:
    import einops
except ImportError:
    missing_deps.append("einops==0.7.*")
try:
    import jaxtyping
except ImportError:
    missing_deps.append("jaxtyping~=0.2.24")

if missing_deps:
    print(f"警告: 缺少以下依赖: {', '.join(missing_deps)}")
    print("请运行: pip install " + " ".join(missing_deps))

# 方式1: 从 uni2ts.model.moirai 导入
try:
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from uni2ts.distribution import NormalOutput
    MOIRAI_AVAILABLE = True
    print("✓ 成功导入 MOIRAI 模块")
except ImportError as e:
    import_errors.append(f"方式1失败: {e}")
    
    # 方式2: 尝试直接导入 uni2ts
    try:
        import uni2ts
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        from uni2ts.distribution import NormalOutput
        MOIRAI_AVAILABLE = True
        print("✓ 成功导入 MOIRAI 模块（方式2）")
    except ImportError as e2:
        import_errors.append(f"方式2失败: {e2}")
        
        # 方式3: 尝试从安装的包导入
        try:
            # 检查是否已安装 uni2ts
            import importlib.util
            spec = importlib.util.find_spec("uni2ts")
            if spec is not None:
                from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
                from uni2ts.distribution import NormalOutput
                MOIRAI_AVAILABLE = True
                print("✓ 成功导入 MOIRAI 模块（方式3：从已安装包）")
            else:
                raise ImportError("uni2ts 包未找到")
        except ImportError as e3:
            import_errors.append(f"方式3失败: {e3}")

if not MOIRAI_AVAILABLE:
    print("=" * 70)
    print("警告: 无法导入 MOIRAI 模块")
    print("=" * 70)
    if missing_deps:
        print(f"\n缺少依赖: {', '.join(missing_deps)}")
        print(f"\n请先安装缺失的依赖:")
        print(f"   pip install {' '.join(missing_deps)}")
    print("\n详细错误信息:")
    for i, error in enumerate(import_errors, 1):
        print(f"  导入方式 {i}: {error}")
    print("\n完整安装步骤：")
    print("\n1. 安装所有依赖:")
    print("   pip install lightning>=2.0 gluonts~=0.14.3 einops==0.7.*")
    print("   pip install jaxtyping~=0.2.24 hydra-core==1.3 huggingface_hub>=0.23.0 safetensors")
    print("\n2. 安装 uni2ts:")
    print("   cd /root/shengyuan/Distillation/uni2ts")
    print("   pip install -e .")
    print("\n3. 验证安装:")
    print("   python -c \"import sys; sys.path.insert(0, 'src'); from uni2ts.model.moirai import MoiraiForecast; print('OK')\"")
    print("=" * 70)


class MoiraiDistillationDataset(Dataset):
    """MOIRAI 蒸馏数据集 - 单变量预测，720预测96"""
    
    def __init__(
        self,
        data_dir: str,
        context_length: int = 720,
        prediction_length: int = 96,
        stride: int = 1,
        scale: bool = True,
        split: str = 'train',
        train_split: float = 0.8,
        test_split: float = 0.1,
        time_col_name: str = 'date',
        max_files: int = 100,
        max_samples_per_file: int = 1000,
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.scale = scale
        self.split = split
        self.time_col_name = time_col_name
        self.max_files = max_files
        self.max_samples_per_file = max_samples_per_file
        self.train_split = train_split
        self.test_split = test_split
        
        # 存储数据段和样本偏移
        self.data_segments = []  # (ts_values, scaler, file_name)
        self.sample_offsets = []  # (segment_idx, start_idx)
        
        # 加载数据
        self._load_data(data_dir)
        
        print(f"数据集加载完成: {len(self.sample_offsets)} 个样本")
    
    def _load_data(self, data_dir: str):
        """加载 CSV 文件并创建样本"""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise ValueError(f"数据目录不存在: {data_dir}")
        
        csv_files = list(data_dir.glob("*.csv"))
        if len(csv_files) == 0:
            raise ValueError(f"在 {data_dir} 中未找到 CSV 文件")
        
        print(f"找到 {len(csv_files)} 个 CSV 文件")
        
        # 限制文件数量
        if len(csv_files) > self.max_files:
            print(f"限制处理前 {self.max_files} 个文件")
            csv_files = csv_files[:self.max_files]
        
        # 计算划分边界
        total_files = len(csv_files)
        train_end = int(total_files * self.train_split)
        val_end = int(total_files * (self.train_split + self.test_split))
        
        # 根据 split 选择文件范围
        if self.split == 'train':
            file_range = (0, train_end)
        elif self.split == 'val':
            file_range = (train_end, val_end)
        else:  # test
            file_range = (val_end, total_files)
        
        csv_files = csv_files[file_range[0]:file_range[1]]
        print(f"使用 {len(csv_files)} 个文件用于 {self.split} 集")
        
        # 处理每个文件
        for file_idx, csv_file in enumerate(csv_files):
            if (file_idx + 1) % 10 == 0:
                print(f"  处理进度: {file_idx + 1}/{len(csv_files)}")
            
            try:
                df = pd.read_csv(csv_file)
                
                # 确定目标列（单变量）
                exclude_cols = ['date', 'timestamp', 'time', 'id', 'item_id', 'time_idx']
                numeric_cols = [col for col in df.columns 
                               if col.lower() not in [c.lower() for c in exclude_cols]
                               and pd.api.types.is_numeric_dtype(df[col])]
                
                if len(numeric_cols) == 0:
                    # 如果没有数值列，使用最后一列
                    numeric_cols = [df.columns[-1]]
                
                # 使用第一个数值列（单变量预测）
                ts_values = df[numeric_cols[0]].values.astype(np.float32)
                
                # 移除 NaN 和 Inf
                valid_mask = ~(np.isnan(ts_values) | np.isinf(ts_values))
                ts_values = ts_values[valid_mask]
                
                if len(ts_values) < self.context_length + self.prediction_length:
                    continue
                
                # 标准化
                scaler = None
                if self.scale:
                    scaler = StandardScaler()
                    ts_values = scaler.fit_transform(ts_values.reshape(-1, 1)).flatten()
                
                # 创建样本
                segment_idx = len(self.data_segments)
                self.data_segments.append((ts_values, scaler, csv_file.name))
                
                # 生成样本偏移
                num_samples = (len(ts_values) - self.context_length - self.prediction_length) // self.stride + 1
                num_samples = min(num_samples, self.max_samples_per_file)
                
                for i in range(0, num_samples * self.stride, self.stride):
                    start_idx = i
                    if start_idx + self.context_length + self.prediction_length > len(ts_values):
                        break
                    self.sample_offsets.append((segment_idx, start_idx))
                
            except Exception as e:
                print(f"警告: 处理文件 {csv_file.name} 时出错: {e}")
                continue
    
    def __len__(self):
        return len(self.sample_offsets)
    
    def __getitem__(self, idx):
        segment_idx, start_idx = self.sample_offsets[idx]
        ts_values, scaler, file_name = self.data_segments[segment_idx]
        
        # 提取上下文和预测序列
        context = ts_values[start_idx:start_idx + self.context_length]
        target = ts_values[start_idx + self.context_length:start_idx + self.context_length + self.prediction_length]
        
        # 转换为 torch tensor
        context = torch.FloatTensor(context)
        target = torch.FloatTensor(target)
        
        return {
            'context': context,  # [context_length]
            'target': target,    # [prediction_length]
            'file_name': file_name
        }


class SimpleMoiraiStudent(nn.Module):
    """简化的 MOIRAI 学生模型"""
    
    def __init__(
        self,
        context_length: int = 720,
        prediction_length: int = 96,
        d_model: int = 128,  # 较小的隐藏维度
        num_layers: int = 4,  # 较少的层数
        num_heads: int = 8,
        dropout: float = 0.1,
        patch_size: int = 16,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_model = d_model
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_size, d_model)
        
        # Positional encoding
        max_seq_len = (context_length // patch_size) + (prediction_length // patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 预测头
        self.predict_head = nn.Linear(d_model, patch_size)
        
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [batch_size, context_length]
        Returns:
            predictions: [batch_size, prediction_length]
        """
        batch_size = context.shape[0]
        
        # 将上下文序列分割为 patch
        context_patches = context.unfold(1, self.patch_size, self.patch_size)  # [B, num_patches, patch_size]
        num_context_patches = context_patches.shape[1]
        
        # Patch embedding
        context_emb = self.patch_embed(context_patches)  # [B, num_patches, d_model]
        
        # 添加位置编码
        context_emb = context_emb + self.pos_embed[:, :num_context_patches, :]
        
        # Transformer 编码
        encoded = self.transformer(context_emb)  # [B, num_patches, d_model]
        
        # 预测未来 patch
        num_pred_patches = (self.prediction_length + self.patch_size - 1) // self.patch_size
        
        # 使用最后一个 patch 的特征来预测所有未来 patch
        # 可以尝试使用平均池化或最后一个 hidden state
        last_hidden = encoded[:, -1:, :]  # [B, 1, d_model]
        
        # 简单的线性投影来生成预测 patch 的 hidden states
        # 使用一个小的 MLP 来生成多个预测位置的 hidden states
        pred_hidden_list = []
        for i in range(num_pred_patches):
            # 每个预测位置使用不同的位置编码
            pos_emb = self.pos_embed[:, num_context_patches + i:num_context_patches + i + 1, :]
            # 使用最后一个 hidden state 加上位置编码
            pred_hidden = last_hidden + pos_emb
            pred_hidden_list.append(pred_hidden)
        
        pred_hidden = torch.cat(pred_hidden_list, dim=1)  # [B, num_pred_patches, d_model]
        
        # 预测 patch
        pred_patches = self.predict_head(pred_hidden)  # [B, num_pred_patches, patch_size]
        
        # 展平并截断到 prediction_length
        predictions = pred_patches.reshape(batch_size, -1)[:, :self.prediction_length]
        
        return predictions


class MoiraiDistillationTrainer:
    """MOIRAI 知识蒸馏训练器"""
    
    def __init__(
        self,
        teacher_model: Any,  # MoiraiForecast
        student_model: SimpleMoiraiStudent,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        alpha: float = 0.5,  # 蒸馏损失权重
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.alpha = alpha
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # 移动模型到设备
        self.teacher_model = self.teacher_model.to(device)
        self.student_model = self.student_model.to(device)
        
        # 冻结教师模型
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        self.best_val_loss = float('inf')
    
    def _teacher_predict(self, context: torch.Tensor) -> torch.Tensor:
        """使用教师模型进行预测"""
        batch_size = context.shape[0]
        context_length = context.shape[1]
        
        # 准备 MOIRAI 输入格式
        # MOIRAI 需要 past_target, past_observed_target, past_is_pad
        # past_target: [B, past_length, target_dim]
        # 注意：past_length 可能包含 prediction_length（如果 patch_size="auto"）
        
        # 获取 past_length（MOIRAI 的属性）
        past_length = self.teacher_model.past_length
        
        # 如果 context_length 小于 past_length，需要 padding
        if context_length < past_length:
            pad_length = past_length - context_length
            past_target = F.pad(context.unsqueeze(-1), (0, 0, 0, pad_length), value=0.0)
            past_observed_target = F.pad(
                torch.ones(batch_size, context_length, 1, dtype=torch.bool, device=context.device),
                (0, 0, 0, pad_length), value=False
            )
        else:
            # 截断到 past_length
            past_target = context[:, :past_length].unsqueeze(-1)  # [B, past_length, 1]
            past_observed_target = torch.ones(batch_size, past_length, 1, dtype=torch.bool, device=context.device)
        
        past_is_pad = torch.zeros(batch_size, past_length, dtype=torch.bool, device=context.device)
        
        with torch.no_grad():
            try:
                # 使用 MOIRAI 的 forward 方法
                predictions = self.teacher_model(
                    past_target=past_target,
                    past_observed_target=past_observed_target,
                    past_is_pad=past_is_pad,
                    num_samples=1  # 只采样一次
                )  # [B, 1, prediction_length, 1] 或 [B, prediction_length]
                
                # 提取预测值（处理不同的输出形状）
                if predictions.dim() == 4:
                    predictions = predictions.squeeze(1).squeeze(-1)  # [B, prediction_length]
                elif predictions.dim() == 3:
                    predictions = predictions.squeeze(-1)  # [B, prediction_length]
                elif predictions.dim() == 2:
                    pass  # 已经是 [B, prediction_length]
                
            except Exception as e:
                print(f"教师模型预测出错: {e}")
                # 返回零预测作为后备
                predictions = torch.zeros(
                    batch_size, 
                    self.teacher_model.hparams.prediction_length,
                    device=context.device
                )
        
        return predictions
    
    def train_epoch(self, epoch: int):
        """训练一个 epoch"""
        self.student_model.train()
        total_loss = 0.0
        total_mse_loss = 0.0
        total_distill_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            context = batch['context'].to(self.device)  # [B, context_length]
            target = batch['target'].to(self.device)   # [B, prediction_length]
            
            # 梯度累积
            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            try:
                # 学生模型预测
                student_pred = self.student_model(context)  # [B, prediction_length]
                
                # 教师模型预测
                teacher_pred = self._teacher_predict(context)  # [B, prediction_length]
                
                # 确保形状匹配
                if teacher_pred.shape != student_pred.shape:
                    min_len = min(teacher_pred.shape[1], student_pred.shape[1])
                    teacher_pred = teacher_pred[:, :min_len]
                    student_pred = student_pred[:, :min_len]
                    target = target[:, :min_len]
                
                # 检查 NaN/Inf
                if torch.isnan(student_pred).any() or torch.isinf(student_pred).any():
                    print(f"警告: Batch {batch_idx + 1} 学生模型输出包含 NaN/Inf，跳过")
                    continue
                
                if torch.isnan(teacher_pred).any() or torch.isinf(teacher_pred).any():
                    print(f"警告: Batch {batch_idx + 1} 教师模型输出包含 NaN/Inf，跳过")
                    continue
                
                # 计算损失
                mse_loss = F.mse_loss(student_pred, target)
                distillation_loss = F.mse_loss(student_pred, teacher_pred)
                
                # 组合损失
                loss = self.alpha * distillation_loss + (1 - self.alpha) * mse_loss
                
                # 检查损失
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                    print(f"警告: Batch {batch_idx + 1} 损失异常，跳过")
                    continue
                
                # 反向传播
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                # 梯度累积
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                
                # 统计
                total_loss += loss.item() * self.gradient_accumulation_steps
                total_mse_loss += mse_loss.item()
                total_distill_loss += distillation_loss.item()
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                    'mse': f'{mse_loss.item():.4f}',
                    'distill': f'{distillation_loss.item():.4f}'
                })
                
            except Exception as e:
                print(f"警告: Batch {batch_idx + 1} 训练出错: {e}")
                import traceback
                traceback.print_exc()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_mse_loss = total_mse_loss / num_batches if num_batches > 0 else 0.0
        avg_distill_loss = total_distill_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'mse_loss': avg_mse_loss,
            'distill_loss': avg_distill_loss
        }
    
    def validate(self):
        """验证"""
        self.student_model.eval()
        total_loss = 0.0
        total_mse_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                context = batch['context'].to(self.device)
                target = batch['target'].to(self.device)
                
                try:
                    student_pred = self.student_model(context)
                    
                    if student_pred.shape != target.shape:
                        min_len = min(student_pred.shape[1], target.shape[1])
                        student_pred = student_pred[:, :min_len]
                        target = target[:, :min_len]
                    
                    mse_loss = F.mse_loss(student_pred, target)
                    total_loss += mse_loss.item()
                    total_mse_loss += mse_loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"验证出错: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return {'val_loss': avg_loss, 'val_mse_loss': avg_loss}
    
    def train(self, num_epochs: int, save_dir: str = './checkpoints/moirai_distill'):
        """训练主循环"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"开始训练，共 {num_epochs} 个 epoch")
        print(f"模型保存目录: {save_dir}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            print(f"训练损失: {train_metrics['loss']:.4f} "
                  f"(MSE: {train_metrics['mse_loss']:.4f}, "
                  f"Distill: {train_metrics['distill_loss']:.4f})")
            
            # 验证
            val_metrics = self.validate()
            print(f"验证损失: {val_metrics['val_loss']:.4f}")
            
            # 学习率调度
            self.scheduler.step(val_metrics['val_loss'])
            
            # 保存最佳模型
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                model_path = save_dir / f'best_model_epoch_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['val_loss'],
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                }, model_path)
                print(f"保存最佳模型: {model_path}")
        
        print(f"\n训练完成！最佳验证损失: {self.best_val_loss:.4f}")


def load_teacher_model(model_name: str = "Salesforce/moirai-1.1-R-base", device: str = 'cuda') -> Any:
    """加载预训练的 MOIRAI 教师模型"""
    if not MOIRAI_AVAILABLE:
        error_msg = """
MOIRAI 模块导入失败！

请按照以下步骤解决：

1. 安装 uni2ts 及其依赖:
   cd /root/shengyuan/Distillation/uni2ts
   pip install -e .

2. 如果安装失败，手动安装依赖:
   pip install lightning>=2.0 gluonts~=0.14.3 einops==0.7.* jaxtyping~=0.2.24
   pip install hydra-core==1.3 huggingface_hub>=0.23.0 safetensors

3. 验证安装:
   python -c "import sys; sys.path.insert(0, 'src'); from uni2ts.model.moirai import MoiraiForecast; print('OK')"

4. 如果仍有问题，请检查:
   - Python 版本 >= 3.10
   - 所有依赖是否正确安装
   - 查看上面的详细错误信息
   pip install torch lightning gluonts einops jaxtyping huggingface_hub

4. 验证安装:
   python -c "from uni2ts.model.moirai import MoiraiForecast; print('OK')"
"""
        raise ImportError(error_msg)
    if MoiraiForecast is None or MoiraiModule is None:
        raise ImportError("MOIRAI 模块未正确导入，请检查导入路径")
    
    print(f"加载教师模型: {model_name}")
    
    try:
        # 从预训练模型加载
        teacher_module = MoiraiModule.from_pretrained(model_name)
        
        # 创建 MoiraiForecast
        teacher_model = MoiraiForecast(
            prediction_length=96,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=720,
            module=teacher_module,
            patch_size="auto",
            num_samples=1,
        )
        
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        
        print("教师模型加载成功")
        return teacher_model
        
    except Exception as e:
        print(f"加载教师模型失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='MOIRAI 知识蒸馏')
    parser.add_argument('--data_dir', type=str, 
                       default='data/datasets/Pretrain_Data',
                       help='数据目录路径')
    parser.add_argument('--context_length', type=int, default=720,
                       help='上下文长度')
    parser.add_argument('--prediction_length', type=int, default=96,
                       help='预测长度')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='蒸馏损失权重')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--teacher_model', type=str, 
                       default='Salesforce/moirai-1.1-R-base',
                       help='教师模型名称')
    parser.add_argument('--save_dir', type=str, 
                       default='./checkpoints/moirai_distill',
                       help='模型保存目录')
    parser.add_argument('--max_files', type=int, default=100,
                       help='最大文件数量')
    
    args = parser.parse_args()
    
    # 检查数据目录
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = script_dir / data_dir
    if not data_dir.exists():
        raise ValueError(f"数据目录不存在: {data_dir}")
    
    print(f"数据目录: {data_dir}")
    print(f"上下文长度: {args.context_length}")
    print(f"预测长度: {args.prediction_length}")
    
    # 创建数据集
    print("\n创建数据集...")
    train_dataset = MoiraiDistillationDataset(
        data_dir=str(data_dir),
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        split='train',
        max_files=args.max_files,
    )
    
    val_dataset = MoiraiDistillationDataset(
        data_dir=str(data_dir),
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        split='val',
        max_files=args.max_files,
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # 加载教师模型
    print("\n加载教师模型...")
    teacher_model = load_teacher_model(args.teacher_model, args.device)
    
    # 创建学生模型
    print("\n创建学生模型...")
    student_model = SimpleMoiraiStudent(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        d_model=384,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        patch_size=16,
    )
    
    print(f"学生模型参数量: {sum(p.numel() for p in student_model.parameters()):,}")
    
    # 创建训练器
    trainer = MoiraiDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
    )
    
    # 开始训练
    trainer.train(num_epochs=args.num_epochs, save_dir=args.save_dir)


if __name__ == '__main__':
    main()

