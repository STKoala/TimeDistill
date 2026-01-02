# Chronos-2 蒸馏方案

本目录提供了使用 Hugging Face TRL 库对 Chronos-2 模型进行知识蒸馏的可行方案。

## 方案概述

由于 Chronos-2 使用自定义架构（非标准 Transformer）和特殊的时间序列 tokenization，直接使用 TRL 的 GKDTrainer 可能存在兼容性问题。我们提供了两个方案：

### 方案 1：手动蒸馏训练（推荐） - `chronos2_distill_gkd.py`

**优点：**
- 完全兼容 Chronos-2 的架构
- 可以精确控制蒸馏过程
- 稳定可靠

**特点：**
- 使用自定义的 `Chronos2DistillationTrainer`
- 实现 KL 散度 + 交叉熵的蒸馏损失
- 支持软标签和硬标签的混合训练

### 方案 2：TRL GKD 适配（实验性） - `chronos2_distill.py`

**优点：**
- 使用标准的 TRL 框架
- 支持 On-policy 蒸馏

**限制：**
- 需要适配 Chronos-2 的特殊 tokenization
- 可能需要额外的适配层

## 快速开始

### 环境准备

```bash
pip install trl transformers datasets accelerate chronos pandas numpy torch
```

### 使用方案 1（推荐）

```bash
python chronos2_distill_gkd.py
```

### 使用方案 2（实验性）

```bash
python chronos2_distill.py
```

## 配置参数

### 教师模型选择

```python
teacher_model_id = "amazon/chronos-2-base"  # 较小版本
# 或
teacher_model_id = "amazon/chronos-2"      # 完整版本
```

### 学生模型配置

通过 `student_config_overrides` 参数控制学生模型大小：

```python
student_config_overrides = {
    "num_layers": 3,      # 减少层数（教师可能是 6 或 12 层）
    "d_model": 384,       # 减少隐藏层维度（教师可能是 512 或 768）
    "d_ff": 1536,         # 减少前馈网络维度
    "num_heads": 6,       # 减少注意力头数
}
```

### 训练参数

```python
# 蒸馏参数
temperature = 2.0    # Logits 平滑温度
alpha = 0.5          # 蒸馏损失权重（0.0=只用硬标签，1.0=只用软标签）

# 训练参数
learning_rate = 5e-5
batch_size = 32
num_epochs = 10
```

### 数据配置

```python
data_paths = [
    "datasets/ETTh1.csv",
    "datasets/ETTh2.csv",
    "datasets/ETTm1.csv",
    "datasets/ETTm2.csv"
]
context_length = 96   # 上下文长度
horizon = 24          # 预测长度
stride = 1            # 滑动窗口步长
target_col = "OT"     # 目标列名
```

## 关键技术细节

### 1. 为什么选择手动蒸馏而不是直接使用 GKD？

Chronos-2 使用：
- 自定义的模型架构（基于 T5 但高度定制）
- 特殊的 tokenization（将时间序列数值直接映射到 embedding，而非文本 token）
- 非常小的词汇表（vocab_size=2）

这些特性使得直接使用 TRL GKD 需要大量适配工作。

### 2. 蒸馏损失函数

方案 1 使用组合损失：

```python
loss = α * KL_divergence(soft_student, soft_teacher) + (1-α) * CrossEntropy(student, hard_labels)
```

其中：
- `α` 控制软标签和硬标签的权重
- `temperature` 控制软标签的平滑程度

### 3. 学生模型初始化

学生模型可以从教师模型初始化部分权重（例如 embedding 层），以加速收敛。当前实现中，学生模型是随机初始化的，可以根据需要添加权重初始化策略。

## 预期效果

- **模型大小**：学生模型通常比教师模型小 50-80%
- **性能**：在大多数任务上，学生模型可以达到教师模型 90-95% 的性能
- **推理速度**：由于模型更小，推理速度可以提升 2-5 倍

## 注意事项

1. **内存需求**：蒸馏训练需要同时加载教师和学生模型，确保有足够的 GPU 内存
2. **数据格式**：确保数据文件包含目标列（默认 "OT"）
3. **模型兼容性**：不同版本的 Chronos-2 模型可能有不同的配置，需要相应调整

## 故障排除

### 问题 1：教师模型预测失败

**原因**：Chronos-2 的 predict 方法需要特定的输入格式

**解决**：检查输入数据格式，确保符合 Chronos-2 的要求

### 问题 2：学生模型前向传播失败

**原因**：Chronos2Model 的 forward 方法可能需要特定的输入格式

**解决**：查看 Chronos-2 的文档，了解正确的输入格式

### 问题 3：内存不足

**解决**：
- 减小 batch_size
- 使用梯度累积
- 使用更小的学生模型

## 进阶使用

### 自定义损失函数

可以修改 `compute_distillation_loss` 方法来实现自定义的蒸馏损失：

```python
def compute_distillation_loss(self, student_logits, teacher_logits, labels):
    # 自定义损失计算
    ...
```

### 权重初始化策略

在 `create_student_model` 函数中添加权重初始化：

```python
# 从教师模型复制 embedding 权重
student_model.shared.weight.data = teacher_model.shared.weight.data[:student_vocab_size]
```

## 参考资源

- [Chronos-2 官方文档](https://github.com/amazon-science/chronos-forecasting)
- [TRL GKD 文档](https://huggingface.co/docs/trl/main/en/gkd_trainer)
- [知识蒸馏论文](https://arxiv.org/abs/1503.02531)

## 许可证

本代码遵循原项目的许可证。

