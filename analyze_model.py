"""
分析学生模型的参数用量、推理时间并生成可视化
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
import seaborn as sns
from pathlib import Path
from chronos import Chronos2Pipeline
import time
import json
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_student_model(model_path: str, device: str = "cuda"):
    """加载学生模型"""
    print(f"正在加载学生模型: {model_path}")
    
    if not os.path.exists(model_path):
        possible_paths = [
            model_path,
            os.path.join(model_path, "pytorch_model.bin"),
            os.path.join(model_path, "model.safetensors"),
            "./chronos-2-distilled/final_model",
            "./chronos-2-distilled",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        else:
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
    
    try:
        student_pipeline = Chronos2Pipeline.from_pretrained(
            model_path,
            device_map=device
        )
        print("成功加载模型")
    except Exception as e:
        print(f"无法作为 pipeline 加载，尝试作为模型加载: {e}")
        try:
            from transformers import AutoConfig
            from chronos import Chronos2Model
            config = AutoConfig.from_pretrained(model_path)
            student_model = Chronos2Model.from_pretrained(model_path)
            student_pipeline = Chronos2Pipeline(student_model)
            student_pipeline.model = student_pipeline.model.to(device)
            print("成功作为模型加载并创建 pipeline")
        except Exception as e2:
            print(f"加载失败: {e2}")
            raise
    
    return student_pipeline


def count_parameters(model):
    """统计模型参数"""
    total_params = 0
    trainable_params = 0
    layer_info = []
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        
        # 按层分组统计
        layer_name = name.split('.')[0] if '.' in name else name
        layer_info.append({
            'name': name,
            'layer': layer_name,
            'params': num_params,
            'trainable': param.requires_grad,
            'shape': list(param.shape)
        })
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
        'layer_info': layer_info
    }


def analyze_model_structure(pipeline):
    """分析模型结构"""
    model = pipeline.model
    config = model.config
    
    print("\n" + "="*60)
    print("模型配置信息")
    print("="*60)
    print(f"模型类型: {config.model_type}")
    print(f"d_model: {config.d_model}")
    print(f"d_kv: {config.d_kv}")
    print(f"d_ff: {config.d_ff}")
    print(f"num_layers: {config.num_layers}")
    print(f"num_heads: {config.num_heads}")
    print(f"dropout_rate: {config.dropout_rate}")
    
    if hasattr(config, 'chronos_config'):
        chronos_config = config.chronos_config
        print(f"\nChronos 配置:")
        print(f"  context_length: {chronos_config.get('context_length', 'N/A')}")
        print(f"  input_patch_size: {chronos_config.get('input_patch_size', 'N/A')}")
        print(f"  output_patch_size: {chronos_config.get('output_patch_size', 'N/A')}")
    
    # 统计参数
    param_info = count_parameters(model)
    
    print("\n" + "="*60)
    print("参数统计")
    print("="*60)
    print(f"总参数数量: {param_info['total']:,} ({param_info['total']/1e6:.2f}M)")
    print(f"可训练参数: {param_info['trainable']:,} ({param_info['trainable']/1e6:.2f}M)")
    print(f"不可训练参数: {param_info['non_trainable']:,} ({param_info['non_trainable']/1e6:.2f}M)")
    
    # 按层分组统计
    layer_params = {}
    for info in param_info['layer_info']:
        layer = info['layer']
        if layer not in layer_params:
            layer_params[layer] = {'total': 0, 'trainable': 0, 'layers': []}
        layer_params[layer]['total'] += info['params']
        if info['trainable']:
            layer_params[layer]['trainable'] += info['params']
        layer_params[layer]['layers'].append(info)
    
    print("\n" + "="*60)
    print("各层参数统计")
    print("="*60)
    for layer, stats in sorted(layer_params.items(), key=lambda x: -x[1]['total']):
        print(f"{layer:30s}: {stats['total']:>12,} ({stats['total']/1e6:>6.2f}M) "
              f"[可训练: {stats['trainable']:>12,}]")
    
    return {
        'config': config,
        'param_info': param_info,
        'layer_params': layer_params
    }


def measure_inference_time(pipeline, context_length=96, horizon=24, num_runs=50, warmup=10):
    """测量推理时间"""
    print("\n" + "="*60)
    print("推理时间测试")
    print("="*60)
    
    device = next(pipeline.model.parameters()).device
    model = pipeline.model
    model.eval()
    
    # 创建测试数据
    test_data = np.random.randn(context_length).astype(np.float32)
    context_df = pd.DataFrame({
        'id': ['test_series'] * context_length,
        'target': test_data
    })
    
    # 预热
    print(f"预热 {warmup} 次...")
    with torch.no_grad():
        for _ in range(warmup):
            try:
                _ = pipeline.predict_df(
                    context_df,
                    prediction_length=horizon,
                    quantile_levels=[0.5],
                    id_column="id",
                    timestamp_column=None,
                    target="target",
                )
            except:
                pass
    
    # 测量推理时间
    print(f"测量 {num_runs} 次推理时间 (context_length={context_length}, horizon={horizon})...")
    inference_times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            try:
                _ = pipeline.predict_df(
                    context_df,
                    prediction_length=horizon,
                    quantile_levels=[0.5],
                    id_column="id",
                    timestamp_column=None,
                    target="target",
                )
            except Exception as e:
                print(f"推理失败: {e}")
                continue
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
            
            if (i + 1) % 10 == 0:
                print(f"  完成 {i+1}/{num_runs} 次")
    
    if len(inference_times) == 0:
        print("警告: 没有成功的推理")
        return None
    
    inference_times = np.array(inference_times)
    
    stats = {
        'mean': np.mean(inference_times),
        'std': np.std(inference_times),
        'min': np.min(inference_times),
        'max': np.max(inference_times),
        'median': np.median(inference_times),
        'p95': np.percentile(inference_times, 95),
        'p99': np.percentile(inference_times, 99),
        'all_times': inference_times.tolist()
    }
    
    print(f"\n推理时间统计 (毫秒):")
    print(f"  平均: {stats['mean']:.2f} ms")
    print(f"  中位数: {stats['median']:.2f} ms")
    print(f"  标准差: {stats['std']:.2f} ms")
    print(f"  最小值: {stats['min']:.2f} ms")
    print(f"  最大值: {stats['max']:.2f} ms")
    print(f"  P95: {stats['p95']:.2f} ms")
    print(f"  P99: {stats['p99']:.2f} ms")
    print(f"  吞吐量: {1000/stats['mean']:.2f} 次/秒")
    
    return stats


def visualize_model_analysis(analysis_result, inference_stats, output_dir="model_analysis"):
    """可视化模型分析结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    param_info = analysis_result['param_info']
    layer_params = analysis_result['layer_params']
    
    # 1. 参数分布饼图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1.1 各层参数分布（饼图）
    ax1 = axes[0, 0]
    layer_names = []
    layer_sizes = []
    for layer, stats in sorted(layer_params.items(), key=lambda x: -x[1]['total']):
        layer_names.append(layer)
        layer_sizes.append(stats['total'])
    
    # 只显示前10个最大的层
    if len(layer_names) > 10:
        top_10 = sorted(zip(layer_names, layer_sizes), key=lambda x: -x[1])[:10]
        layer_names = [x[0] for x in top_10]
        layer_sizes = [x[1] for x in top_10]
        other_size = sum([x[1] for x in sorted(zip(layer_names, layer_sizes), key=lambda x: -x[1])[10:]])
        if other_size > 0:
            layer_names.append('其他')
            layer_sizes.append(other_size)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(layer_names)))
    wedges, texts, autotexts = ax1.pie(layer_sizes, labels=layer_names, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax1.set_title('各层参数分布', fontsize=14, fontweight='bold')
    
    # 1.2 各层参数柱状图
    ax2 = axes[0, 1]
    layer_names_sorted = sorted(layer_params.keys(), key=lambda x: -layer_params[x]['total'])
    layer_sizes_sorted = [layer_params[l]['total'] / 1e6 for l in layer_names_sorted]  # 转换为百万
    
    bars = ax2.barh(range(len(layer_names_sorted)), layer_sizes_sorted, color='steelblue')
    ax2.set_yticks(range(len(layer_names_sorted)))
    ax2.set_yticklabels(layer_names_sorted)
    ax2.set_xlabel('参数数量 (百万)', fontsize=12)
    ax2.set_title('各层参数数量', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, size) in enumerate(zip(bars, layer_sizes_sorted)):
        ax2.text(size, i, f'{size:.2f}M', va='center', ha='left', fontsize=9)
    
    # 1.3 推理时间分布
    ax3 = axes[1, 0]
    if inference_stats and 'all_times' in inference_stats:
        times = inference_stats['all_times']
        ax3.hist(times, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax3.axvline(inference_stats['mean'], color='red', linestyle='--', linewidth=2, label=f'平均值: {inference_stats["mean"]:.2f}ms')
        ax3.axvline(inference_stats['median'], color='green', linestyle='--', linewidth=2, label=f'中位数: {inference_stats["median"]:.2f}ms')
        ax3.set_xlabel('推理时间 (毫秒)', fontsize=12)
        ax3.set_ylabel('频次', fontsize=12)
        ax3.set_title('推理时间分布', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
    
    # 1.4 参数统计总结
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
模型参数统计总结

总参数数量: {param_info['total']:,}
({param_info['total']/1e6:.2f}M)

可训练参数: {param_info['trainable']:,}
({param_info['trainable']/1e6:.2f}M)

不可训练参数: {param_info['non_trainable']:,}
({param_info['non_trainable']/1e6:.2f}M)

主要层数: {len(layer_params)}
"""
    if inference_stats:
        summary_text += f"""
推理性能:
  平均时间: {inference_stats['mean']:.2f} ms
  中位数: {inference_stats['median']:.2f} ms
  吞吐量: {1000/inference_stats['mean']:.2f} 次/秒
"""
    
    ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_analysis_overview.png'), dpi=300, bbox_inches='tight')
    print(f"\n已保存概览图: {os.path.join(output_dir, 'model_analysis_overview.png')}")
    plt.close()
    
    # 2. 详细的层参数分析
    fig, ax = plt.subplots(figsize=(14, max(8, len(layer_names_sorted) * 0.4)))
    
    # 创建更详细的条形图，显示可训练和不可训练参数
    trainable_sizes = [layer_params[l]['trainable'] / 1e6 for l in layer_names_sorted]
    non_trainable_sizes = [(layer_params[l]['total'] - layer_params[l]['trainable']) / 1e6 
                           for l in layer_names_sorted]
    
    x = np.arange(len(layer_names_sorted))
    width = 0.6
    
    bars1 = ax.barh(x, trainable_sizes, width, label='可训练参数', color='steelblue')
    bars2 = ax.barh(x, non_trainable_sizes, width, left=trainable_sizes, 
                    label='不可训练参数', color='lightcoral')
    
    ax.set_yticks(x)
    ax.set_yticklabels(layer_names_sorted)
    ax.set_xlabel('参数数量 (百万)', fontsize=12)
    ax.set_title('各层参数详细分布（可训练 vs 不可训练）', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # 添加总数值标签
    for i, (train, non_train) in enumerate(zip(trainable_sizes, non_trainable_sizes)):
        total = train + non_train
        ax.text(total, i, f'{total:.2f}M', va='center', ha='left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_parameters_detail.png'), dpi=300, bbox_inches='tight')
    print(f"已保存详细参数图: {os.path.join(output_dir, 'layer_parameters_detail.png')}")
    plt.close()
    
    # 3. 推理时间箱线图（如果有多次测试）
    if inference_stats and len(inference_stats.get('all_times', [])) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times = inference_stats['all_times']
        bp = ax.boxplot([times], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
        
        ax.set_ylabel('推理时间 (毫秒)', fontsize=12)
        ax.set_title('推理时间箱线图', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 添加统计信息
        stats_text = f"平均值: {inference_stats['mean']:.2f} ms\n"
        stats_text += f"中位数: {inference_stats['median']:.2f} ms\n"
        stats_text += f"标准差: {inference_stats['std']:.2f} ms\n"
        stats_text += f"P95: {inference_stats['p95']:.2f} ms\n"
        stats_text += f"吞吐量: {1000/inference_stats['mean']:.2f} 次/秒"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               family='monospace', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'inference_time_boxplot.png'), dpi=300, bbox_inches='tight')
        print(f"已保存推理时间箱线图: {os.path.join(output_dir, 'inference_time_boxplot.png')}")
        plt.close()


def save_analysis_report(analysis_result, inference_stats, output_dir="model_analysis"):
    """保存分析报告为 JSON"""
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        'model_config': {
            'd_model': analysis_result['config'].d_model,
            'd_kv': analysis_result['config'].d_kv,
            'd_ff': analysis_result['config'].d_ff,
            'num_layers': analysis_result['config'].num_layers,
            'num_heads': analysis_result['config'].num_heads,
            'dropout_rate': analysis_result['config'].dropout_rate,
        },
        'parameters': {
            'total': analysis_result['param_info']['total'],
            'trainable': analysis_result['param_info']['trainable'],
            'non_trainable': analysis_result['param_info']['non_trainable'],
        },
        'layer_parameters': {
            layer: {
                'total': stats['total'],
                'trainable': stats['trainable'],
                'non_trainable': stats['total'] - stats['trainable']
            }
            for layer, stats in analysis_result['layer_params'].items()
        },
        'inference_stats': inference_stats
    }
    
    report_path = os.path.join(output_dir, 'analysis_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"已保存分析报告: {report_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='分析学生模型的参数用量和推理时间')
    parser.add_argument('--model_path', type=str, default='./chronos-2-distilled/final_model',
                        help='学生模型路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--context_length', type=int, default=96,
                        help='测试时的上下文长度')
    parser.add_argument('--horizon', type=int, default=24,
                        help='测试时的预测长度')
    parser.add_argument('--num_runs', type=int, default=50,
                        help='推理时间测试次数')
    parser.add_argument('--warmup', type=int, default=10,
                        help='预热次数')
    parser.add_argument('--output_dir', type=str, default='model_analysis',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        args.device = 'cpu'
    
    print("="*60)
    print("学生模型分析工具")
    print("="*60)
    
    # 加载模型
    try:
        pipeline = load_student_model(args.model_path, args.device)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 分析模型结构
    analysis_result = analyze_model_structure(pipeline)
    
    # 测量推理时间
    inference_stats = measure_inference_time(
        pipeline,
        context_length=args.context_length,
        horizon=args.horizon,
        num_runs=args.num_runs,
        warmup=args.warmup
    )
    
    # 生成可视化
    print("\n" + "="*60)
    print("生成可视化图表")
    print("="*60)
    visualize_model_analysis(analysis_result, inference_stats, args.output_dir)
    
    # 保存报告
    save_analysis_report(analysis_result, inference_stats, args.output_dir)
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
    print(f"所有结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()

