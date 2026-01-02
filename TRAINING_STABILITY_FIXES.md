# 训练稳定性修复说明

## 发现的问题

训练过程中出现了严重的数值不稳定问题：
- 损失值出现巨大波动（如 `8693482495410176.0000`）
- 某些 batch 的损失值正常，某些突然爆炸
- 这是典型的**梯度爆炸**和**数值不稳定**问题

## 根本原因

1. **缺少梯度裁剪** - 梯度爆炸导致损失急剧增大
2. **数据范围差异大** - Monash 数据集中不同文件的数据范围差异很大
3. **没有 NaN/Inf 检测** - 异常值没有被检测和处理
4. **学习率可能过大** - 5e-5 对于这种大规模数据可能不稳定
5. **缺少损失值限制** - 异常 batch 导致损失爆炸

## 已实施的修复

### 1. **梯度裁剪**
```python
# 梯度裁剪，防止梯度爆炸
grad_norm = torch.nn.utils.clip_grad_norm_(
    self.student_model.parameters(), 
    max_grad_norm=1.0
)
```

### 2. **NaN/Inf 检测和处理**
```python
# 检查模型输出
if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
    print(f"警告: Batch {batch_idx + 1} 学生模型输出包含 NaN/Inf，跳过")
    continue

# 检查损失值
if torch.isnan(loss) or torch.isinf(loss):
    continue

# 检查梯度
for param in self.student_model.parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            # 跳过更新
            continue
```

### 3. **损失值限制**
```python
# 限制损失值，防止数值爆炸
if loss.item() > 1e6:
    print(f"警告: Batch {batch_idx + 1} 损失过大，跳过")
    continue
```

### 4. **降低学习率**
```python
learning_rate = 1e-5  # 从 5e-5 降低到 1e-5
```

### 5. **优化器改进**
```python
self.optimizer = torch.optim.AdamW(
    self.student_model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4,  # 添加权重衰减
    eps=1e-8  # 数值稳定性
)
```

### 6. **数据验证**
```python
# 检查数据有效性
if np.isinf(ts_values).any():
    print(f"跳过: 包含 Inf 值")
    continue

# 检查数值范围
if max_val > 1e10:
    print(f"警告: 数值范围很大，可能影响训练")
```

## 预期效果

修复后应该看到：
- ✅ 损失值更加稳定，不会出现突然的爆炸
- ✅ 梯度范数被限制在合理范围内
- ✅ 异常 batch 会被自动跳过，不会影响训练
- ✅ 训练过程更加平滑

## 如果问题仍然存在

如果训练仍然不稳定，可以尝试：

1. **进一步降低学习率**
   ```python
   learning_rate = 5e-6  # 更小的学习率
   ```

2. **减小 batch size**
   ```python
   batch_size = 16  # 或 8
   ```

3. **使用学习率调度器**
   ```python
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, mode='min', factor=0.5, patience=5
   )
   ```

4. **增加梯度裁剪阈值**
   ```python
   max_grad_norm = 0.5  # 更严格的梯度裁剪
   ```

5. **数据预处理**
   - 对每个数据集进行归一化
   - 移除异常值
   - 使用更严格的数据过滤

## 监控指标

训练时注意观察：
- **损失值**：应该平滑下降，不应有突然的跳跃
- **梯度范数**：应该保持在 1.0 以下
- **警告信息**：如果频繁出现 NaN/Inf 警告，需要进一步调查

## 总结

主要修复：
1. ✅ 添加梯度裁剪
2. ✅ 添加 NaN/Inf 检测
3. ✅ 降低学习率
4. ✅ 添加损失值限制
5. ✅ 改进优化器配置
6. ✅ 增强数据验证

这些修复应该能显著提高训练稳定性！

