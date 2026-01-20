# MOIRAI 知识蒸馏

基于 uni2ts 中的 MOIRAI 源码实现的知识蒸馏脚本，用于单变量时间序列预测任务。

## 功能特点

- **单变量预测**：支持单变量时间序列预测
- **720预测96**：上下文长度720，预测长度96
- **知识蒸馏**：使用预训练的 MOIRAI 模型作为教师模型，训练轻量级学生模型
- **数据集支持**：使用 `TimeDistill/data/datasets/Pretrain_Data` 中的 CSV 数据集

## 环境要求

1. 安装 uni2ts：
```bash
cd uni2ts
pip install -e .
```

2. 安装其他依赖：
```bash
pip install torch torchvision pandas numpy scikit-learn tqdm
```

## 使用方法

### 基本用法

```bash
python moirai_distill.py \
    --data_dir data/datasets/Pretrain_Data \
    --context_length 720 \
    --prediction_length 96 \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --alpha 0.5 \
    --device cuda
```

### 参数说明

- `--data_dir`: 数据目录路径（默认：`data/datasets/Pretrain_Data`）
- `--context_length`: 上下文长度（默认：720）
- `--prediction_length`: 预测长度（默认：96）
- `--batch_size`: 批次大小（默认：32）
- `--num_epochs`: 训练轮数（默认：10）
- `--learning_rate`: 学习率（默认：1e-4）
- `--alpha`: 蒸馏损失权重，范围 [0, 1]（默认：0.5）
  - `alpha=1.0`: 只使用蒸馏损失（软标签）
  - `alpha=0.0`: 只使用真实标签损失（硬标签）
  - `alpha=0.5`: 平衡两种损失
- `--device`: 设备（默认：cuda）
- `--teacher_model`: 教师模型名称（默认：`Salesforce/moirai-1.1-R-base`）
- `--save_dir`: 模型保存目录（默认：`./checkpoints/moirai_distill`）
- `--max_files`: 最大处理文件数量（默认：100）

## 模型架构

### 教师模型
- 使用预训练的 MOIRAI 模型（`Salesforce/moirai-1.1-R-base`）
- 模型参数冻结，仅用于生成软标签

### 学生模型
- **简化版 MOIRAI**：轻量级 Transformer 架构
- **参数配置**：
  - `d_model`: 128（隐藏维度）
  - `num_layers`: 4（Transformer 层数）
  - `num_heads`: 8（注意力头数）
  - `patch_size`: 16（patch 大小）
- **参数量**：约 1-2M（相比教师模型大幅减少）

## 数据集格式

数据集应为 CSV 格式，包含：
- **时间列**：`date`、`timestamp`、`time` 等（会被自动识别并忽略）
- **数值列**：至少一个数值列用于预测（单变量）

示例：
```csv
date,T2
2017-01-01 14:00:00,467.0
2017-01-02 14:00:00,443.0
...
```

## 训练流程

1. **数据加载**：
   - 从指定目录加载所有 CSV 文件
   - 自动识别数值列（单变量）
   - 数据标准化（StandardScaler）
   - 按 8:1:1 划分训练/验证/测试集

2. **模型初始化**：
   - 加载预训练的 MOIRAI 教师模型
   - 创建简化的学生模型
   - 冻结教师模型参数

3. **训练循环**：
   - 对每个批次：
     - 学生模型预测
     - 教师模型预测（生成软标签）
     - 计算损失：
       - 硬标签损失：`MSE(student_pred, target)`
       - 软标签损失：`MSE(student_pred, teacher_pred)`
       - 组合损失：`alpha * soft_loss + (1 - alpha) * hard_loss`
   - 反向传播和参数更新
   - 梯度裁剪（max_grad_norm=1.0）

4. **模型保存**：
   - 每个 epoch 验证后保存最佳模型
   - 保存路径：`{save_dir}/best_model_epoch_{epoch}.pt`

## 输出文件

训练完成后，会在 `save_dir` 目录下生成：
- `best_model_epoch_{epoch}.pt`: 最佳模型检查点
  - 包含模型状态、优化器状态、训练指标等

## 注意事项

1. **内存使用**：
   - 教师模型较大，建议使用 GPU
   - 如果内存不足，可以减小 `batch_size` 或 `max_files`

2. **数据质量**：
   - 脚本会自动处理 NaN 和 Inf 值
   - 如果数据质量较差，可能需要预处理

3. **训练稳定性**：
   - 包含 NaN/Inf 检查
   - 梯度裁剪防止梯度爆炸
   - 学习率调度器自动调整学习率

4. **模型兼容性**：
   - 确保 uni2ts 已正确安装
   - 确保可以访问 HuggingFace Hub（或使用镜像）

## 示例输出

```
数据目录: /path/to/data/datasets/Pretrain_Data
上下文长度: 720
预测长度: 96

创建数据集...
找到 100 个 CSV 文件
使用 80 个文件用于 train 集
数据集加载完成: 5000 个样本

加载教师模型...
加载教师模型: Salesforce/moirai-1.1-R-base
教师模型加载成功

创建学生模型...
学生模型参数量: 1,234,567

开始训练，共 10 个 epoch
模型保存目录: ./checkpoints/moirai_distill

Epoch 1/10
训练损失: 0.1234 (MSE: 0.1500, Distill: 0.1000)
验证损失: 0.1100
保存最佳模型: ./checkpoints/moirai_distill/best_model_epoch_1.pt
...
```

## 故障排除

1. **导入错误**：
   - 确保 uni2ts 已正确安装
   - 检查 Python 路径设置

2. **CUDA 内存不足**：
   - 减小 `batch_size`
   - 减小 `max_files`
   - 使用 CPU（`--device cpu`）

3. **数据加载失败**：
   - 检查数据目录路径
   - 确保 CSV 文件格式正确
   - 检查是否有足够的有效数据

4. **教师模型加载失败**：
   - 检查网络连接（HuggingFace Hub）
   - 使用镜像：`export HF_ENDPOINT=https://hf-mirror.com`

## 参考

- [uni2ts GitHub](https://github.com/SalesforceAIResearch/uni2ts)
- [MOIRAI Paper](https://arxiv.org/abs/2402.02592)

