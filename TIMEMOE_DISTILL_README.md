# TimeMoE 知识蒸馏使用指南

本文档介绍如何在 TimeDistill 目录中使用 TimeMoE 知识蒸馏框架。

## 文件结构

```
TimeDistill/
├── timemoe_distill.py          # 主蒸馏脚本
├── TIMEMOE_DISTILL_README.md   # 本文档
└── timemoe-distilled/          # 输出目录（训练后生成）
    ├── logs/                   # 日志文件
    ├── tb_logs/                # TensorBoard 日志
    └── ...                     # 模型检查点
```

## 前置条件

1. **Time-MoE 代码库**：确保 `Time-MoE` 目录在 `Distillation` 目录下，与 `TimeDistill` 同级
2. **数据**：确保 `data/datasets/Pretrain_Data` 目录存在并包含 CSV 数据文件
3. **依赖**：安装必要的 Python 包（参考 Time-MoE 的 requirements.txt）

## 快速开始

### 基本使用

```bash
cd TimeDistill
python timemoe_distill.py
```

### 默认配置

- **数据目录**: `data/datasets/Pretrain_Data`
- **教师模型**: `Maple728/TimeMoE-50M`
- **输出目录**: `./timemoe-distilled`
- **输入长度**: 720
- **预测长度**: 96
- **学生模型**: 2层，hidden_size=128，2个专家

## 配置说明

在 `timemoe_distill.py` 的 `main()` 函数中可以修改以下参数：

### 数据配置

```python
data_dir = find_data_dir()  # 自动查找 data/datasets/Pretrain_Data
max_length = 720            # 输入序列长度
horizon = 96                # 预测长度
stride = None               # 滑动窗口步长（None=等于max_length）
normalization_method = "zero"  # 归一化方法（zero/max/none）
```

### 学生模型配置

```python
student_config_overrides = {
    'hidden_size': 128,         # 隐藏层大小
    'num_hidden_layers': 2,     # 层数
    'num_attention_heads': 4,   # 注意力头数
    'intermediate_size': 256,   # 中间层大小
    'num_experts': 2,           # 专家数
    'num_experts_per_tok': 1,   # 每个token选择的专家数
    'input_size': 1,            # 输入维度（单变量=1）
}
```

### 蒸馏参数

```python
alpha = 1.0      # 输出蒸馏权重（必须 > 0）
beta = 0.5       # 隐层蒸馏权重（0=禁用）
gamma = 0.1      # 路由蒸馏权重（0=禁用）
temperature = 1.0    # 温度参数
lambda_gt = 0.0      # 真实标签损失权重（0=纯蒸馏，1=纯监督）
use_hook = False     # 是否使用 Hook 提取特征
hook_layers = [0, -1]  # Hook 层索引
```

### 训练参数

```python
learning_rate = 1e-4           # 学习率
min_learning_rate = 5e-5       # 最小学习率
global_batch_size = 64         # 全局batch size
micro_batch_size = 16          # 每个设备的batch size
num_train_epochs = 10          # 训练轮数
precision = "bf16"             # 精度（fp32/fp16/bf16）
gradient_checkpointing = False # 梯度检查点（节省内存）
weight_decay = 0.1             # 权重衰减
warmup_ratio = 0.1             # Warmup比例
lr_scheduler_type = "cosine"   # 学习率调度器
max_grad_norm = 1.0            # 梯度裁剪
logging_steps = 100            # 日志步数
save_steps = 1000              # 保存步数
seed = 9899                    # 随机种子
```

## 输出说明

### 目录结构

训练完成后，会在 `output_dir` 下生成：

```
timemoe-distilled/
├── logs/
│   └── timemoe_distill_YYYYMMDD_HHMMSS.log  # 训练日志
├── tb_logs/                                  # TensorBoard 日志
├── config.json                               # 模型配置
├── pytorch_model.bin                         # 模型权重
└── checkpoint-*/                             # 检查点目录（如果启用）
```

### 日志内容

日志文件记录：
- 模型配置信息
- 参数量统计
- 训练进度和损失
- 错误和警告信息

### TensorBoard

```bash
tensorboard --logdir timemoe-distilled/tb_logs
```

## 数据格式要求

数据文件应为 CSV 格式，包含：
- 一个时间列（如 `date`, `timestamp`）
- 一个或多个数值列（作为目标变量）

示例：
```csv
date,T1537
2012-01-26 00:00:01,0.11
2012-01-26 01:00:01,0.13
...
```

脚本会自动：
- 识别时间列并忽略
- 使用第一个数值列作为目标变量
- 处理缺失值

## 常见问题

### 1. 找不到 Time-MoE 模块

**错误**: `ImportError: No module named 'time_moe'`

**解决**: 确保 `Time-MoE` 目录在正确位置，与 `TimeDistill` 同级：
```
Distillation/
├── Time-MoE/
└── TimeDistill/
```

### 2. HuggingFace 模型下载失败（429 错误或网络问题）

**错误**: 
```
429 Client Error: Too Many Requests
或
We couldn't connect to 'https://hf-mirror.com' to load the files
```

**解决方案**:

#### 方案 A: 使用本地模型路径（推荐）

1. **下载模型到本地**:
   ```bash
   # 使用 huggingface-cli
   huggingface-cli download Maple728/TimeMoE-50M --local-dir ./models/time_moe_50m
   
   # 或使用 Python
   from transformers import AutoModel
   model = AutoModel.from_pretrained("Maple728/TimeMoE-50M")
   model.save_pretrained("./models/time_moe_50m")
   ```

2. **修改代码使用本地路径**:
   ```python
   # 在 timemoe_distill.py 的 main() 函数中
   teacher_model_path = "./models/time_moe_50m"  # 使用本地路径
   ```

#### 方案 B: 使用其他 HuggingFace 镜像

修改环境变量或代码：
```python
# 在代码开头
os.environ["HF_ENDPOINT"] = "https://huggingface.co"  # 使用官方镜像
# 或
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 使用镜像站
```

#### 方案 C: 使用已缓存的模型

如果模型已下载到缓存目录，可以尝试：
```python
# 代码会自动检测本地路径并使用 local_files_only=True
teacher_model_path = "~/.cache/huggingface/hub/models--Maple728--TimeMoE-50M/snapshots/..."
```

### 3. 找不到数据目录

**错误**: `错误: 未找到数据目录 Pretrain_Data`

**解决**: 确保数据目录位于以下位置之一：
- `TimeDistill/data/datasets/Pretrain_Data`
- `./data/datasets/Pretrain_Data`

或在代码中修改 `find_data_dir()` 函数添加自定义路径。

### 4. 显存不足

**解决**:
- 减小 `micro_batch_size`
- 启用 `gradient_checkpointing = True`
- 使用 `precision = "fp16"` 或 `precision = "bf16"`
- 减小 `max_length`

### 5. 训练不稳定

**解决**:
- 降低学习率
- 先只用输出蒸馏（`beta=0, gamma=0`）
- 检查数据归一化
- 减小 `max_length`

### 6. torch_dtype 警告

**警告**: `torch_dtype is deprecated! Use dtype instead!`

**状态**: 已在代码中修复，使用 `dtype` 参数。如果仍看到警告，请更新到最新版本代码。

## 进阶使用

### 使用 Hook 提取特征

```python
use_hook = True
hook_layers = [0, -1]  # 提取第1层和最后1层的特征
```

### 混合训练（蒸馏+监督）

```python
lambda_gt = 0.2  # 20% 真实标签损失，80% 蒸馏损失
```

### 使用本地模型路径

如果网络连接有问题，可以使用本地模型：

```python
# 在 timemoe_distill.py 的 main() 函数中
teacher_model_path = "./models/time_moe_50m"  # 本地模型路径
```

确保本地路径包含：
- `config.json`
- `pytorch_model.bin` 或 `model.safetensors`
- 其他必要的模型文件

### 自定义数据目录

修改 `find_data_dir()` 函数或直接设置：

```python
data_dir = "/path/to/your/data/directory"
```

## 性能建议

1. **单变量预测**: 使用 `input_size=1`，`max_length=720`，`horizon=96`
2. **批量大小**: `global_batch_size >= 64` 效果更好
3. **学习率**: 学生模型通常需要较大学习率（1e-4 到 5e-4）
4. **蒸馏权重**: 推荐 `alpha=1.0, beta=0.5, gamma=0.1`

## 参考

- Time-MoE 原始代码: `../Time-MoE/`
- 蒸馏框架文档: `../Time-MoE/README_DISTILL.md`
- 其他蒸馏脚本: `timesmoe_distill_gkd.py`, `chronos2_distill_gkd.py`

