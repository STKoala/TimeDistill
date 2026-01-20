# 配置系统使用说明

## 概述

本项目采用**参数选项 + 配置类**的工程结构，支持：
- 从YAML配置文件加载配置
- 使用命令行参数覆盖配置
- 类型安全的配置类（使用dataclass）

## 配置结构

配置系统采用分层设计，主要包含以下配置类：

- `Chronos2DistillConfig`: 顶层配置类，包含所有子配置
  - `ModelConfig`: 模型配置（教师模型ID、学生模型参数等）
  - `DataConfig`: 数据配置（数据路径、上下文长度、预测长度等）
  - `TrainingConfig`: 训练配置（学习率、批次大小、训练轮数等）
  - `LossConfig`: 损失函数配置（各种损失权重、DTW参数等）
  - `SystemConfig`: 系统配置（设备、输出目录、日志等）

## 使用方法

### 1. 使用YAML配置文件

创建或编辑 `configs/chronos-2-distill.yaml`：

```yaml
model:
  teacher_model_id: "amazon/chronos-2"
  student_config_overrides:
    num_layers: 6
    d_model: 384
    d_ff: 1536
    num_heads: 6

data:
  data_dir: "data/datasets/Pretrain_Data"
  context_length: 720
  horizon: 96
  stride: 96

training:
  learning_rate: 1e-5
  batch_size: 256
  num_epochs: 20
```

然后运行：

```bash
python -m TimeDistill.scripts.chronos2_distill_feature_dtw --config configs/chronos-2-distill.yaml
```

### 2. 使用命令行参数覆盖配置

可以在使用YAML配置文件的同时，用命令行参数覆盖部分配置：

```bash
python -m TimeDistill.scripts.chronos2_distill_feature_dtw \
  --config configs/chronos-2-distill.yaml \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --num_epochs 30
```

### 3. 完全使用命令行参数（不使用YAML）

如果不指定 `--config`，将使用默认配置，然后可以用命令行参数覆盖：

```bash
python -m TimeDistill.scripts.chronos2_distill_feature_dtw \
  --teacher_model_id "amazon/chronos-2-base" \
  --data_dir "data/datasets/Pretrain_Data" \
  --batch_size 128 \
  --learning_rate 1e-5
```

### 4. 在代码中使用配置

```python
from TimeDistill.configs import load_config, create_arg_parser

# 解析命令行参数
parser = create_arg_parser()
args = parser.parse_args()

# 加载配置
config = load_config(config_path=args.config, args=args)

# 使用配置
print(f"教师模型: {config.model.teacher_model_id}")
print(f"学习率: {config.training.learning_rate}")
print(f"批次大小: {config.training.batch_size}")
```

## 配置项说明

### ModelConfig（模型配置）

- `teacher_model_id`: 教师模型ID，如 "amazon/chronos-2"
- `student_config_overrides`: 学生模型配置覆盖，字典格式

### DataConfig（数据配置）

- `data_dir`: 数据目录路径
- `dataset_names`: 数据集名称列表，None表示加载目录下所有CSV文件
- `context_length`: 上下文长度
- `horizon`: 预测长度
- `stride`: 滑动窗口步长
- `target_col`: 目标列名称，None表示自动检测
- `train_split`: 训练集比例
- `test_split`: 测试集比例
- `scale`: 是否标准化数据
- `time_col_name`: 时间列名称

### TrainingConfig（训练配置）

- `learning_rate`: 学习率
- `batch_size`: 批次大小
- `num_epochs`: 训练轮数
- `max_grad_norm`: 梯度裁剪阈值
- `num_workers`: DataLoader工作进程数（GPU训练建议设为0）
- `pin_memory`: 是否使用pin_memory

### LossConfig（损失函数配置）

- `alpha_pred_distill`: 预测蒸馏权重（0-1之间，0.5表示soft和hard各占一半）
- `beta_feature`: 特征蒸馏权重
- `dtw_weight`: DTW损失权重（0表示禁用）
- `dtw_gamma`: DTW损失gamma参数
- `use_fast_dtw`: 是否使用快速DTW近似
- `temperature`: 温度参数（预留）

### SystemConfig（系统配置）

- `device`: 设备（如 "cuda:0", "cpu"）
- `output_dir`: 输出目录
- `verbose`: 是否打印详细信息
- `hf_endpoint`: HuggingFace镜像端点
- `cuda_visible_devices`: CUDA可见设备

## 命令行参数列表

所有配置项都可以通过命令行参数覆盖，参数名与配置项路径对应：

- `--config`: YAML配置文件路径
- `--teacher_model_id`: 教师模型ID
- `--data_dir`: 数据目录
- `--dataset_names`: 数据集名称列表（空格分隔）
- `--context_length`: 上下文长度
- `--horizon`: 预测长度
- `--stride`: 滑动窗口步长
- `--learning_rate`: 学习率
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数
- `--max_grad_norm`: 梯度裁剪阈值
- `--alpha_pred_distill`: 预测蒸馏权重
- `--beta_feature`: 特征蒸馏权重
- `--dtw_weight`: DTW损失权重
- `--dtw_gamma`: DTW损失gamma参数
- `--device`: 设备
- `--output_dir`: 输出目录
- `--verbose`: 是否打印详细信息

## 最佳实践

1. **使用YAML配置文件管理实验配置**：为不同的实验创建不同的YAML配置文件
2. **使用命令行参数进行快速调参**：在YAML基础上用命令行参数覆盖部分配置
3. **版本控制配置文件**：将YAML配置文件纳入版本控制，便于复现实验
4. **使用默认配置进行开发**：开发时可以使用默认配置，生产环境使用YAML文件

## 示例

### 示例1：基础训练

```bash
python -m TimeDistill.scripts.chronos2_distill_feature_dtw \
  --config configs/chronos-2-distill.yaml
```

### 示例2：快速实验（小批次、少轮数）

```bash
python -m TimeDistill.scripts.chronos2_distill_feature_dtw \
  --config configs/chronos-2-distill.yaml \
  --batch_size 64 \
  --num_epochs 5
```

### 示例3：调整损失权重

```bash
python -m TimeDistill.scripts.chronos2_distill_feature_dtw \
  --config configs/chronos-2-distill.yaml \
  --alpha_pred_distill 0.7 \
  --beta_feature 0.2 \
  --dtw_weight 0.1
```

