# 配置系统迁移说明

## 概述

已将代码从硬编码参数方式迁移到**参数选项 + 配置类**的工程结构，提供了更灵活、可维护的配置管理方式。

## 主要变更

### 1. 新增文件

- `configs/config.py`: 定义所有配置的dataclass类
  - `Chronos2DistillConfig`: 顶层配置类
  - `ModelConfig`: 模型配置
  - `DataConfig`: 数据配置
  - `TrainingConfig`: 训练配置
  - `LossConfig`: 损失函数配置
  - `SystemConfig`: 系统配置

- `configs/config_loader.py`: 配置加载器
  - `load_config_from_yaml()`: 从YAML文件加载配置
  - `merge_config_with_args()`: 用命令行参数覆盖配置
  - `create_arg_parser()`: 创建命令行参数解析器
  - `load_config()`: 统一的配置加载入口

- `configs/chronos-2-distill.yaml`: 示例YAML配置文件

- `configs/__init__.py`: 配置模块导出

- `configs/README.md`: 配置系统使用文档

### 2. 修改文件

- `scripts/chronos2_distill_feature_dtw.py`: 重构为使用配置类系统
  - 移除了硬编码的参数
  - 添加了命令行参数解析
  - 支持从YAML配置文件加载
  - 支持命令行参数覆盖

## 使用方式对比

### 旧方式（硬编码参数）

```python
def main(
    teacher_model_id: str = "amazon/chronos-2",
    data_dir: str = "data/datasets/Pretrain_Data",
    ...
):
    # 所有参数都在函数签名中硬编码
    ...
```

### 新方式（配置类 + YAML + 命令行）

```python
# 1. 使用YAML配置文件
python -m TimeDistill.scripts.chronos2_distill_feature_dtw --config configs/chronos-2-distill.yaml

# 2. 使用YAML + 命令行覆盖
python -m TimeDistill.scripts.chronos2_distill_feature_dtw \
  --config configs/chronos-2-distill.yaml \
  --batch_size 128 \
  --learning_rate 2e-5

# 3. 完全使用命令行参数
python -m TimeDistill.scripts.chronos2_distill_feature_dtw \
  --teacher_model_id "amazon/chronos-2" \
  --data_dir "data/datasets/Pretrain_Data" \
  --batch_size 256
```

## 配置结构

```
Chronos2DistillConfig
├── ModelConfig
│   ├── teacher_model_id
│   └── student_config_overrides
├── DataConfig
│   ├── data_dir
│   ├── dataset_names
│   ├── context_length
│   ├── horizon
│   └── ...
├── TrainingConfig
│   ├── learning_rate
│   ├── batch_size
│   ├── num_epochs
│   └── ...
├── LossConfig
│   ├── alpha_pred_distill
│   ├── beta_feature
│   ├── dtw_weight
│   └── ...
└── SystemConfig
    ├── device
    ├── output_dir
    ├── verbose
    └── ...
```

## 优势

1. **类型安全**: 使用dataclass提供类型提示和验证
2. **灵活配置**: 支持YAML文件、命令行参数、默认值三种方式
3. **易于维护**: 配置集中管理，便于版本控制
4. **易于扩展**: 新增配置项只需在dataclass中添加字段
5. **向后兼容**: 保留了所有原有功能，只是改变了配置方式

## 迁移检查清单

- [x] 创建配置类系统
- [x] 创建配置加载器
- [x] 创建YAML配置文件
- [x] 重构训练脚本
- [x] 创建使用文档
- [x] 测试配置加载功能

## 后续建议

1. **其他脚本迁移**: 可以考虑将其他训练脚本（如 `chronos2_distill_gkd_feature_and_dtw.py`）也迁移到配置类系统
2. **配置验证**: 可以添加配置验证逻辑，确保配置值的合理性
3. **配置模板**: 可以为不同场景创建多个配置模板（如快速实验、完整训练等）
4. **配置继承**: 可以实现配置继承机制，支持基础配置和实验特定配置

## 测试

配置系统已通过基本测试：

```bash
# 测试导入
python -c "from TimeDistill.configs import Chronos2DistillConfig, load_config; print('OK')"

# 测试YAML加载
python -c "from TimeDistill.configs import load_config_from_yaml; config = load_config_from_yaml('TimeDistill/configs/chronos-2-distill.yaml'); print(f'教师模型: {config.model.teacher_model_id}')"
```

## 注意事项

1. YAML文件中的 `null` 值需要用 `null` 关键字（不是 `None`）
2. 命令行参数会覆盖YAML配置文件中的对应项
3. 如果不指定 `--config`，将使用默认配置
4. 配置文件的路径可以是绝对路径或相对于项目根目录的相对路径

