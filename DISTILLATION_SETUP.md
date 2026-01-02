# Chronos-2 蒸馏配置说明

## 数据配置

### 训练数据
- **来源**: `datasets/monash_csv_downsmp/` 目录下的所有 CSV 文件
- **自动处理**: 脚本会自动：
  - 扫描目录下所有 `.csv` 文件
  - 自动识别目标列（排除 date/timestamp 等列后的第一个数值列）
  - 处理缺失值和数据长度不足的情况

### 验证数据
- **来源**: `datasets/ETTh1.csv`
- **用途**: 用于验证模型性能，选择最佳模型

### 测试数据
- **来源**: `datasets/ETTh1.csv`
- **用途**: 最终测试和可视化（通过 `test_student_model.py`）

## 使用方法

### 1. 训练蒸馏模型

```bash
python chronos2_distill_gkd.py
```

训练过程：
- 从 `datasets/monash_csv_downsmp/` 加载所有 CSV 文件作为训练数据
- 使用 `datasets/ETTh1.csv` 作为验证数据
- 模型保存在 `./chronos-2-distilled/` 目录

### 2. 测试学生模型

```bash
python test_student_model.py
```

测试过程：
- 加载训练好的学生模型
- 在 `datasets/ETTh1.csv` 上进行预测
- 生成可视化对比图

## 配置参数

在 `chronos2_distill_gkd.py` 的 `main()` 函数中可以修改：

```python
# 训练数据目录
train_data_dir = "datasets/monash_csv_downsmp"

# 验证数据
eval_data_paths = ["datasets/ETTh1.csv"]

# 训练参数
context_length = 96  # 上下文长度
horizon = 24         # 预测长度
stride = 1           # 滑动窗口步长
target_col = None    # None 表示自动检测
```

## 数据格式要求

Monash CSV 文件应包含：
- 至少一个数值列（作为目标变量）
- 可选：date/timestamp 列（会被自动忽略）
- 数据长度应 >= context_length + horizon

## 注意事项

1. **数据量**: Monash 数据集包含大量文件，训练时间可能较长
2. **内存**: 确保有足够内存加载所有数据
3. **验证集**: 使用 ETTh1 作为验证集，确保模型在未见过的数据上表现良好
4. **测试集**: 测试时使用 ETTh1 的不同时间点，避免数据泄露

