# 测试结果汇总和可视化脚本使用说明

## 脚本说明

### 1. `summarize_results.py` - 汇总脚本

该脚本用于汇总 `log/` 目录中最新的 JSON 测试结果到表格（CSV 和 Excel 格式）。

**功能：**
- 自动扫描 `log/` 目录中的所有 JSON 结果文件
- 按数据集和模型类型分组，选择最新的结果
- 生成汇总表格（CSV 和 Excel 格式）
- 显示统计信息

**使用方法：**
```bash
# 使用默认参数（log 目录，输出 summary_results.csv）
python summarize_results.py

# 指定日志目录和输出文件
python summarize_results.py --log_dir log --output summary_results.csv
```

**参数说明：**
- `--log_dir`: 日志目录路径（默认：`log`）
- `--output`: 输出文件名（默认：`summary_results.csv`）

**输出文件：**
- `summary_results.csv`: CSV 格式的汇总表格
- `summary_results.xlsx`: Excel 格式的汇总表格（需要安装 openpyxl）

### 2. `visualize_results.py` - 可视化脚本

该脚本用于可视化展示汇总后的测试结果，生成多种图表。

**功能：**
- 指标对比图（MSE, MAE, RMSE, MAPE）
- 按数据集分组的指标对比
- 热力图展示
- 学生模型相对于教师模型的改进比例
- 汇总统计图

**使用方法：**
```bash
# 生成所有图表（默认）
python visualize_results.py

# 指定输入文件和输出目录
python visualize_results.py --input summary_results.csv --output_dir visualizations

# 只生成特定类型的图表
python visualize_results.py --comparison  # 只生成指标对比图
python visualize_results.py --heatmap    # 只生成热力图
python visualize_results.py --improvement # 只生成改进比例图
```

**参数说明：**
- `--input`: 输入的汇总 CSV 文件（默认：`summary_results.csv`）
- `--output_dir`: 输出目录（默认：`visualizations`）
- `--all`: 生成所有图表（默认行为）
- `--comparison`: 生成指标对比图
- `--by_dataset`: 按数据集生成图表
- `--heatmap`: 生成热力图
- `--improvement`: 生成改进比例图
- `--summary`: 生成汇总统计图

**依赖安装：**
```bash
pip install matplotlib seaborn openpyxl
```

## 完整使用流程

1. **汇总测试结果：**
   ```bash
   cd /root/shengyuan/Distillation/TimeDistill
   python summarize_results.py
   ```

2. **生成可视化图表：**
   ```bash
   python visualize_results.py
   ```

3. **查看结果：**
   - 汇总表格：`summary_results.csv` 或 `summary_results.xlsx`
   - 可视化图表：`visualizations/` 目录下的 PNG 图片

## 输出文件说明

### 汇总表格列说明：
- `model_type`: 模型类型（teacher/student）
- `model_name`: 模型名称
- `dataset`: 数据集名称
- `context_length`: 上下文长度
- `horizon`: 预测长度
- `MSE`: 均方误差
- `MAE`: 平均绝对误差
- `RMSE`: 均方根误差
- `MAPE`: 平均绝对百分比误差
- `total_windows`: 总窗口数
- `successful_windows`: 成功窗口数
- `failed_windows`: 失败窗口数
- `timestamp_str`: 时间戳字符串
- `filename`: 源 JSON 文件名

### 可视化图表说明：
- `metrics_comparison.png`: 四个指标的对比图
- `metrics_<dataset>.png`: 每个数据集的指标对比图
- `heatmap_<metric>.png`: 各指标的热力图
- `improvement_ratio.png`: 学生模型相对于教师模型的改进比例
- `summary_statistics.png`: 模型平均性能对比

## 注意事项

1. 确保 `log/` 目录中有 JSON 格式的测试结果文件
2. JSON 文件命名格式应为：`eval_results_<model_type>_<model_name>_ctx<context>_pred<prediction>_<dataset>_<timestamp>.json`
3. 如果缺少依赖包，请先安装：`pip install matplotlib seaborn openpyxl pandas`
4. 可视化脚本需要先运行汇总脚本生成 CSV 文件

