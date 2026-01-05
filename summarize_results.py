#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总 log/ 目录中最新的 JSON 测试结果到表格
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re


def parse_filename(filename):
    """
    解析 JSON 文件名，提取模型类型、模型名称、数据集和时间戳
    
    格式: eval_results_<model_type>_<model_name>_ctx<context>_pred<prediction>_<dataset>_<timestamp>.json
    """
    basename = os.path.basename(filename)
    if not basename.startswith('eval_results_') or not basename.endswith('.json'):
        return None
    
    # 移除前缀和后缀
    name = basename.replace('eval_results_', '').replace('.json', '')
    
    # 提取时间戳（最后一部分，格式：YYYYMMDD_HHMMSS）
    timestamp_match = re.search(r'(\d{8}_\d{6})$', name)
    if not timestamp_match:
        return None
    
    timestamp_str = timestamp_match.group(1)
    # 移除时间戳部分
    name = name[:timestamp_match.start()].rstrip('_')
    
    # 提取 ctx 和 pred
    ctx_match = re.search(r'ctx(\d+)', name)
    pred_match = re.search(r'pred(\d+)', name)
    
    context_length = int(ctx_match.group(1)) if ctx_match else None
    horizon = int(pred_match.group(1)) if pred_match else None
    
    # 移除 ctx 和 pred 部分
    if ctx_match:
        name = name.replace(ctx_match.group(0), '').strip('_')
    if pred_match:
        name = name.replace(pred_match.group(0), '').strip('_')
    
    # 分割模型类型和模型名称
    parts = name.split('_', 1)
    if len(parts) < 2:
        return None
    
    model_type = parts[0]  # teacher 或 student
    model_name = parts[1] if len(parts) > 1 else ''
    
    # 数据集名称是剩余部分
    dataset = name.replace(f'{model_type}_', '').replace(f'ctx{context_length}_', '').replace(f'pred{horizon}_', '') if context_length and horizon else name.replace(f'{model_type}_', '')
    
    # 解析时间戳
    try:
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    except:
        timestamp = None
    
    return {
        'model_type': model_type,
        'model_name': model_name,
        'context_length': context_length,
        'horizon': horizon,
        'dataset': dataset,
        'timestamp': timestamp,
        'timestamp_str': timestamp_str,
        'filename': filename
    }


def load_json_results(filepath):
    """加载 JSON 结果文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"警告: 无法加载 {filepath}: {e}")
        return None


def get_latest_results(log_dir='log'):
    """
    获取每个数据集和模型类型的最新结果
    
    返回: dict, key 为 (model_type, dataset), value 为最新的结果信息
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"错误: 目录 {log_dir} 不存在")
        return {}
    
    # 存储所有结果
    all_results = defaultdict(list)
    
    # 遍历所有 JSON 文件
    for json_file in log_path.glob('eval_results_*.json'):
        file_info = parse_filename(str(json_file))
        if file_info is None:
            continue
        
        data = load_json_results(json_file)
        if data is None:
            continue
        
        # 从 JSON 数据中获取实际的数据集名称（更准确）
        if 'results' in data and len(data['results']) > 0:
            actual_dataset = data['results'][0].get('dataset', file_info['dataset'])
        else:
            actual_dataset = file_info['dataset']
        
        # 使用 JSON 中的实际值
        model_type = data.get('model_type', file_info['model_type'])
        context_length = data.get('context_length', file_info['context_length'])
        horizon = data.get('horizon', file_info['horizon'])
        
        # 提取模型名称
        if model_type == 'teacher':
            model_name = data.get('teacher_model_id', file_info['model_name'])
        else:
            model_path = data.get('model_path', '')
            if model_path:
                model_name = os.path.basename(model_path.rstrip('/'))
                if not model_name or model_name == '.':
                    model_name = os.path.basename(os.path.dirname(model_path))
            else:
                model_name = file_info['model_name']
        
        # 提取指标
        if 'results' in data and len(data['results']) > 0:
            result = data['results'][0]
            metrics = result.get('metrics', {})
            
            # 创建结果记录
            record = {
                'model_type': model_type,
                'model_name': model_name,
                'dataset': actual_dataset,
                'context_length': context_length,
                'horizon': horizon,
                'timestamp': file_info['timestamp'],
                'timestamp_str': file_info['timestamp_str'],
                'filename': str(json_file),
                'MSE': metrics.get('MSE'),
                'MAE': metrics.get('MAE'),
                'RMSE': metrics.get('RMSE'),
                'MAPE': metrics.get('MAPE'),
                'total_windows': result.get('total_windows'),
                'successful_windows': result.get('successful_windows'),
                'failed_windows': result.get('failed_windows'),
            }
            
            # 使用 (model_type, dataset) 作为 key
            key = (model_type, actual_dataset)
            all_results[key].append(record)
    
    # 对每个 key，选择最新的结果
    latest_results = {}
    for key, records in all_results.items():
        # 按时间戳排序，选择最新的
        records.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min, reverse=True)
        latest_results[key] = records[0]
    
    return latest_results


def create_summary_table(latest_results, output_file='summary_results.csv'):
    """
    创建汇总表格
    
    Parameters
    ----------
    latest_results : dict
        最新的结果字典
    output_file : str
        输出文件名
    """
    if not latest_results:
        print("没有找到任何结果")
        return
    
    # 转换为列表
    records = list(latest_results.values())
    
    # 创建 DataFrame
    df = pd.DataFrame(records)
    
    # 重新排列列的顺序
    columns_order = [
        'model_type',
        'model_name',
        'dataset',
        'context_length',
        'horizon',
        'MSE',
        'MAE',
        'RMSE',
        'MAPE',
        'total_windows',
        'successful_windows',
        'failed_windows',
        'timestamp_str',
        'filename'
    ]
    
    # 只保留存在的列
    columns_order = [col for col in columns_order if col in df.columns]
    df = df[columns_order]
    
    # 按模型类型和数据集排序
    df = df.sort_values(['model_type', 'dataset'])
    
    # 保存为 CSV
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n汇总结果已保存到: {output_file}")
    print(f"共 {len(df)} 条记录")
    
    # 同时保存为 Excel（如果安装了 openpyxl）
    try:
        excel_file = output_file.replace('.csv', '.xlsx')
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"汇总结果已保存到: {excel_file}")
    except ImportError:
        print("提示: 安装 openpyxl 可以同时保存为 Excel 格式: pip install openpyxl")
    except Exception as e:
        print(f"保存 Excel 文件时出错: {e}")
    
    # 显示预览
    print("\n结果预览:")
    print(df.head(20).to_string())
    
    # 显示统计信息
    print("\n统计信息:")
    print(f"模型类型分布:")
    print(df['model_type'].value_counts())
    print(f"\n数据集数量: {df['dataset'].nunique()}")
    
    return df


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='汇总 log/ 目录中最新的 JSON 测试结果')
    parser.add_argument('--log_dir', type=str, default='log', help='日志目录路径')
    parser.add_argument('--output', type=str, default='summary_results.csv', help='输出文件名')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("开始汇总测试结果...")
    print("=" * 60)
    
    # 获取最新结果
    latest_results = get_latest_results(args.log_dir)
    
    if not latest_results:
        print("未找到任何 JSON 结果文件")
        return
    
    print(f"\n找到 {len(latest_results)} 个最新的测试结果")
    
    # 创建汇总表格
    df = create_summary_table(latest_results, args.output)
    
    print("\n" + "=" * 60)
    print("汇总完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

