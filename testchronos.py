import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from chronos import Chronos2Pipeline
import warnings
warnings.filterwarnings('ignore')

# 加载模型
print("正在加载 chronos-2 模型...")
pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
print("模型加载完成")

# 定义测试参数
CONTEXT_LENGTHS = [96, 192, 336]  # 不同的context长度
HORIZON_LENGTHS = [24, 48, 96]     # 不同的horizon长度

def prepare_data_for_chronos(df, start_idx, context_len, target_col='OT'):
    """
    准备chronos-2需要的数据格式
    """
    # 提取context数据
    context_data = df.iloc[start_idx:start_idx+context_len]
    
    # 创建 chronos-2 需要的 DataFrame 格式
    # chronos-2 需要: id, timestamp, target列
    context_df = pd.DataFrame({
        'id': ['series_0'] * len(context_data),
        'timestamp': pd.to_datetime(context_data['date']),
        'target': context_data[target_col].values
    })
    
    return context_df

def predict_with_chronos(pipeline, context_df, horizon_len):
    """
    使用chronos-2进行预测
    """
    try:
        # 使用predict_df方法进行预测
        pred_df = pipeline.predict_df(
            context_df,
            prediction_length=horizon_len,
            quantile_levels=[0.5],  # 只使用中位数预测
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        
        # 提取预测值（中位数）
        # chronos-2返回的DataFrame可能包含多个列，需要找到预测值列
        if '0.5' in pred_df.columns:
            predictions = pred_df['0.5'].values
        elif 'target' in pred_df.columns:
            predictions = pred_df['target'].values
        elif 'mean' in pred_df.columns:
            predictions = pred_df['mean'].values
        else:
            # 如果列名不同，取第一列数值列（排除id和timestamp）
            exclude_cols = ['id', 'timestamp', 'item_id', 'time_idx']
            numeric_cols = [col for col in pred_df.columns 
                          if col not in exclude_cols and 
                          pd.api.types.is_numeric_dtype(pred_df[col])]
            if len(numeric_cols) > 0:
                predictions = pred_df[numeric_cols[0]].values
            else:
                return None
        
        # 确保返回的是1D数组
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        
        return predictions
    except Exception as e:
        print(f"预测出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_dataset(dataset_path, dataset_name):
    """
    测试单个数据集
    """
    print(f"\n正在处理数据集: {dataset_name}")
    
    # 读取数据
    df = pd.read_csv(dataset_path)
    
    # 确保 date 列存在
    if 'date' not in df.columns:
        print(f"警告: {dataset_name} 没有 'date' 列，跳过")
        return []
    
    # 确定目标列（通常是最后一列，或名为'OT'的列）
    if 'OT' in df.columns:
        target_col = 'OT'
    else:
        # 使用最后一列作为目标
        target_col = df.columns[-1]
    
    print(f"使用目标列: {target_col}")
    
    results = []
    total_rows = len(df)
    
    # 遍历不同的context和horizon组合
    for context_len in CONTEXT_LENGTHS:
        for horizon_len in HORIZON_LENGTHS:
            print(f"  测试 context={context_len}, horizon={horizon_len}")
            
            # 使用滑动窗口，步长为horizon_len
            # 确保有足够的数据
            max_start_idx = total_rows - context_len - horizon_len
            
            if max_start_idx < 0:
                print(f"    警告: 数据不足，跳过 (需要至少 {context_len + horizon_len} 行)")
                continue
            
            # 遍历每个窗口（使用滑动窗口，步长为horizon_len）
            window_count = 0
            for start_idx in range(0, max_start_idx + 1, horizon_len):
                try:
                    # 准备context数据
                    context_df = prepare_data_for_chronos(df, start_idx, context_len, target_col)
                    
                    # 进行预测
                    predictions = predict_with_chronos(pipeline, context_df, horizon_len)
                    
                    if predictions is None or len(predictions) == 0:
                        continue
                    
                    # 获取真实值
                    true_values = df[target_col].iloc[start_idx+context_len:start_idx+context_len+horizon_len].values
                    
                    # 确保预测值和真实值长度一致
                    min_len = min(len(predictions), len(true_values))
                    if min_len == 0:
                        continue
                    
                    predictions = predictions[:min_len]
                    true_values = true_values[:min_len]
                    
                    # 计算MSE和MAE
                    mse = mean_squared_error(true_values, predictions)
                    mae = mean_absolute_error(true_values, predictions)
                    
                    # 记录结果
                    results.append({
                        'dataset_name': dataset_name,
                        'row_index': start_idx,
                        'context_length': context_len,
                        'horizon_length': horizon_len,
                        'chronos2_mse': mse,
                        'chronos2_mae': mae
                    })
                    
                    window_count += 1
                    
                    # 每处理100个窗口打印一次进度
                    if window_count % 100 == 0:
                        print(f"    已处理 {window_count} 个窗口...")
                except Exception as e:
                    print(f"    处理窗口 {start_idx} 时出错: {e}")
                    continue
    
    print(f"完成 {dataset_name}，共 {len(results)} 条记录")
    return results

def main():
    """
    主函数：遍历所有数据集并保存结果
    """
    datasets_dir = Path("datasets")
    result_file = "result.csv"
    
    if not datasets_dir.exists():
        print(f"错误: 数据集目录 {datasets_dir} 不存在")
        return
    
    # 获取所有CSV文件
    dataset_files = list(datasets_dir.glob("*.csv"))
    
    if len(dataset_files) == 0:
        print(f"错误: 在 {datasets_dir} 中没有找到CSV文件")
        return
    
    print(f"找到 {len(dataset_files)} 个数据集文件")
    
    all_results = []
    
    # 遍历每个数据集
    for dataset_file in dataset_files:
        dataset_name = dataset_file.stem  # 不带扩展名的文件名
        results = test_dataset(dataset_file, dataset_name)
        all_results.extend(results)
    
    # 转换为DataFrame并保存
    if len(all_results) > 0:
        result_df = pd.DataFrame(all_results)
        
        # 确保列的顺序：数据集名称、行号、context长度、horizon长度、MSE、MAE
        columns_order = ['dataset_name', 'row_index', 'context_length', 'horizon_length', 
                         'chronos2_mse', 'chronos2_mae']
        result_df = result_df[columns_order]
        
        # 保存到CSV
        result_df.to_csv(result_file, index=False)
        print(f"\n结果已保存到 {result_file}")
        print(f"共 {len(result_df)} 条记录")
        print(f"\n结果预览:")
        print(result_df.head(10))
    else:
        print("没有生成任何结果")

if __name__ == "__main__":
    main()
