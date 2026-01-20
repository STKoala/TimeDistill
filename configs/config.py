"""
配置类定义 - 使用 dataclass 定义所有配置项
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ModelConfig:
    """模型配置"""
    teacher_model_id: str = "amazon/chronos-2"
    student_config_overrides: Dict[str, Any] = field(default_factory=lambda: {
        "num_layers": 6,
        "d_model": 384,
        "d_ff": 1536,
        "num_heads": 6,
    })


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "data/datasets/Pretrain_Data"
    dataset_names: Optional[List[str]] = None
    context_length: int = 720
    horizon: int = 96
    stride: int = 96
    target_col: Optional[str] = None
    train_split: float = 0.8
    test_split: float = 0.0
    scale: bool = True
    time_col_name: str = "date"


@dataclass
class TrainingConfig:
    """训练配置"""
    learning_rate: float = 1e-5
    batch_size: int = 256
    num_epochs: int = 20
    max_grad_norm: float = 1.0
    num_workers: int = 0  # GPU训练时建议设为0
    pin_memory: bool = False


@dataclass
class LossConfig:
    """损失函数配置"""
    alpha_pred_distill: float = 0.5  # 预测蒸馏(soft) vs 真实标签(hard) 的插值
    beta_feature: float = 0.3  # 特征蒸馏权重
    dtw_weight: float = 0.2  # DTW 损失权重（0 表示禁用）
    dtw_gamma: float = 1.0
    use_fast_dtw: bool = True
    temperature: float = 2.0


@dataclass
class SystemConfig:
    """系统配置"""
    device: str = "cuda:0"
    output_dir: str = "./chronos-2-distilled"
    verbose: bool = True
    hf_endpoint: str = "https://hf-mirror.com"
    cuda_visible_devices: str = "0"


@dataclass
class Chronos2DistillConfig:
    """Chronos-2 蒸馏完整配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def __post_init__(self):
        """后处理：确保嵌套配置是实例而不是字典"""
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.loss, dict):
            self.loss = LossConfig(**self.loss)
        if isinstance(self.system, dict):
            self.system = SystemConfig(**self.system)

