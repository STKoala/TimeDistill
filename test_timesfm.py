"""
TimesFM 推理测试脚本
从数据目录读取数据集，使用 TimesFM 进行推理并可视化结果
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Union
from sklearn.preprocessing import StandardScaler
import warnings
import json

warnings.filterwarnings('ignore')

# 尝试导入 TimesFM
try:
    import timesfm
    TimesFM_2p5_200M_torch = timesfm.TimesFM_2p5_200M_torch
    ForecastConfig = timesfm.ForecastConfig
    TIMESFM_AVAILABLE = True
except (ImportError, AttributeError) as e:
    TIMESFM_AVAILABLE = False
    print(f"警告: timesfm 模块未正确安装或导入失败: {e}")
    print("请确保已安装: pip install timesfm")

# 尝试导入自定义学生模型
try:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from timesfm_distill_gkd import SimpleTimesFMStudent, create_student_model
    STUDENT_MODEL_AVAILABLE = True
except (ImportError, AttributeError) as e:
    STUDENT_MODEL_AVAILABLE = False
    print(f"警告: 无法导入学生模型类: {e}")


def load_time_series_data(
    data_path: str,
    context_length: int = 720,
    horizon: int = 96,
    target_col: Optional[str] = None,
    time_col_name: str = 'date',
    start_idx: Optional[int] = None
) -> Optional[tuple]:
    """
    加载时间序列数据
    
    Parameters
    ----------
    data_path : str
        数据文件路径
    context_length : int
        上下文长度
    horizon : int
        预测长度
    target_col : str, optional
        目标列名
    time_col_name : str
        时间列名
    start_idx : int, optional
        起始索引（用于测试不同数据段）
        
    Returns
    -------
    tuple or None
        (context_data, future_data, scaler, file_name, original_data, start_idx) 或 None
    """
    if not os.path.exists(data_path):
        print(f"文件不存在: {data_path}")
        return None
    
    try:
        df = pd.read_csv(data_path)
        
        # 确定目标列
        exclude_cols = ['date', 'timestamp', 'time', 'id', 'item_id', 'time_idx']
        numeric_cols = [col for col in df.columns 
                       if col.lower() not in [c.lower() for c in exclude_cols]
                       and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_cols) == 0:
            target_col_actual = df.columns[-1]
        elif target_col and target_col in df.columns:
            target_col_actual = target_col
        else:
            target_col_actual = numeric_cols[0]
        
        # 提取数据
        if time_col_name in df.columns:
            ts_values = df[target_col_actual].values.astype(np.float32)
        else:
            ts_values = df[target_col_actual].values.astype(np.float32)
        
        # 检查数据有效性
        if len(ts_values) < context_length + horizon:
            print(f"数据长度不足: {len(ts_values)} < {context_length + horizon}")
            return None
        
        if np.isinf(ts_values).any() or np.isnan(ts_values).any():
            print(f"数据包含无效值 (Inf 或 NaN)")
            return None
        
        # 如果指定了 start_idx，使用该索引开始的数据段
        if start_idx is not None:
            if start_idx + context_length + horizon > len(ts_values):
                print(f"  警告: start_idx={start_idx} 超出数据范围，跳过")
                return None
            ts_values_segment = ts_values[start_idx:start_idx + context_length + horizon]
        else:
            ts_values_segment = ts_values[:context_length + horizon]
        
        # 标准化数据 - 只使用 context 部分来拟合 scaler（避免数据泄露）
        # 这更符合实际预测场景：我们只能使用历史数据来标准化
        context_raw = ts_values_segment[:context_length]
        future_raw = ts_values_segment[context_length:context_length + horizon]
        
        scaler = StandardScaler()
        scaler.fit(context_raw.reshape(-1, 1))
        
        # 标准化 context 和 future
        context_data = scaler.transform(context_raw.reshape(-1, 1)).flatten()
        future_data = scaler.transform(future_raw.reshape(-1, 1)).flatten()
        
        # 打印数据统计信息用于调试
        print(f"  原始数据统计 - Context: mean={np.mean(context_raw):.4f}, std={np.std(context_raw):.4f}, min={np.min(context_raw):.4f}, max={np.max(context_raw):.4f}")
        print(f"  原始数据统计 - Future: mean={np.mean(future_raw):.4f}, std={np.std(future_raw):.4f}, min={np.min(future_raw):.4f}, max={np.max(future_raw):.4f}")
        print(f"  标准化后统计 - Context: mean={np.mean(context_data):.4f}, std={np.std(context_data):.4f}, min={np.min(context_data):.4f}, max={np.max(context_data):.4f}")
        print(f"  标准化后统计 - Future: mean={np.mean(future_data):.4f}, std={np.std(future_data):.4f}, min={np.min(future_data):.4f}, max={np.max(future_data):.4f}")
        
        file_name = os.path.basename(data_path)
        original_data = ts_values_segment
        actual_start_idx = start_idx if start_idx is not None else 0
        return (context_data, future_data, scaler, file_name, original_data, actual_start_idx)
        
    except Exception as e:
        print(f"加载数据失败 {data_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_student_model(model_path: str, device: str = "cuda"):
    """
    加载自定义的学生模型（SimpleTimesFMStudent）
    
    Parameters
    ----------
    model_path : str
        模型路径（目录路径）
    device : str
        设备
        
    Returns
    -------
    model or None
    """
    if not STUDENT_MODEL_AVAILABLE:
        print("学生模型类不可用")
        return None
    
    print(f"正在加载学生模型: {model_path}")
    
    # 查找模型文件
    if os.path.isdir(model_path):
        model_file = os.path.join(model_path, "pytorch_model.bin")
        config_file = os.path.join(model_path, "config.json")
    else:
        model_file = model_path
        config_file = os.path.join(os.path.dirname(model_path), "config.json")
    
    if not os.path.exists(model_file):
        print(f"错误: 找不到模型文件: {model_file}")
        return None
    
    try:
        # 加载配置
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            context_length = config.get('context_length', 720)
            horizon = config.get('horizon', 96)
            d_model = config.get('d_model', 256)
            nhead = config.get('nhead', 4)
            num_layers = config.get('num_layers', 3)
            dim_feedforward = config.get('dim_feedforward', 512)
            dropout = config.get('dropout', 0.1)
        else:
            # 从检查点加载配置
            checkpoint = torch.load(model_file, map_location='cpu')
            context_length = checkpoint.get('context_length', 720)
            horizon = checkpoint.get('horizon', 96)
            d_model = 256
            nhead = 4
            num_layers = 3
            dim_feedforward = 512
            dropout = 0.1
        
        # 创建模型
        model = SimpleTimesFMStudent(
            context_length=context_length,
            horizon=horizon,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # 加载权重
        checkpoint = torch.load(model_file, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        print(f"学生模型加载完成 (context_length={context_length}, horizon={horizon})")
        return model
    except Exception as e:
        print(f"学生模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_model(model_id: str = "google/timesfm-2.5-200m-pytorch", device: str = "cuda", local_path: Optional[str] = None):
    """
    加载 TimesFM 模型（支持官方模型和自定义学生模型）
    
    Parameters
    ----------
    model_id : str
        模型ID（HuggingFace Hub 上的模型ID）
    device : str
        设备
    local_path : str, optional
        本地模型路径（如果提供，将优先使用本地路径）
        
    Returns
    -------
    model or None
    """
    # 确定要使用的模型路径
    if local_path:
        model_path = local_path
        print(f"正在从本地路径加载模型: {model_path}")
        # 检查路径是否存在
        if not os.path.exists(model_path):
            print(f"错误: 本地模型路径不存在: {model_path}")
            return None
        if not os.path.isdir(model_path):
            print(f"错误: 本地模型路径不是目录: {model_path}")
            return None
        
        # 检查是否是自定义学生模型（有 pytorch_model.bin 文件）
        model_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            print("检测到自定义学生模型格式，使用学生模型加载器")
            return load_student_model(model_path, device)
        
        # 否则尝试作为官方 TimesFM 模型加载
        if not TIMESFM_AVAILABLE:
            print("TimesFM 不可用，无法加载官方模型")
            return None
        
        try:
            model = TimesFM_2p5_200M_torch.from_pretrained(
                model_path,
                torch_compile=True
            )
            
            # 配置模型
            model.compile(
                ForecastConfig(
                    max_context=1024,
                    max_horizon=256,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            
            print("官方 TimesFM 模型加载完成")
            return model
        except Exception as e:
            print(f"作为官方模型加载失败: {e}")
            print("尝试作为学生模型加载...")
            return load_student_model(model_path, device)
    else:
        # 从 HuggingFace Hub 加载官方模型
        if not TIMESFM_AVAILABLE:
            print("TimesFM 不可用")
            return None
        
        print(f"正在从 HuggingFace Hub 加载模型: {model_id}")
        try:
            model = TimesFM_2p5_200M_torch.from_pretrained(
                model_id,
                torch_compile=True
            )
            
            # 配置模型
            model.compile(
                ForecastConfig(
                    max_context=1024,
                    max_horizon=256,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            
            print("模型加载完成")
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None


def predict(model, context_data: np.ndarray, horizon: int = 96):
    """
    使用模型进行预测（支持官方 TimesFM 模型和自定义学生模型）
    
    Parameters
    ----------
    model
        TimesFM 模型或 SimpleTimesFMStudent 模型
    context_data : np.ndarray
        上下文数据，形状为 (context_length,)
    horizon : int
        预测长度
        
    Returns
    -------
    np.ndarray
        预测结果，形状为 (horizon,)
    """
    try:
        # 检查是否是自定义学生模型
        if isinstance(model, nn.Module) and hasattr(model, 'context_length') and hasattr(model, 'horizon'):
            # 自定义学生模型
            model.eval()
            
            # 检查输入长度是否匹配
            model_context_length = model.context_length
            actual_context_length = len(context_data)
            
            if actual_context_length != model_context_length:
                print(f"  警告: 输入长度 ({actual_context_length}) 与模型期望长度 ({model_context_length}) 不匹配")
                if actual_context_length > model_context_length:
                    # 截取最后 model_context_length 个点
                    print(f"  截取最后 {model_context_length} 个时间步")
                    context_data = context_data[-model_context_length:]
                else:
                    # 填充到 model_context_length
                    print(f"  填充到 {model_context_length} 个时间步")
                    padding = np.zeros(model_context_length - actual_context_length)
                    context_data = np.concatenate([padding, context_data])
            
            with torch.no_grad():
                # 转换为 tensor
                context_tensor = torch.FloatTensor(context_data).unsqueeze(0)  # [1, context_length]
                
                # 移动到模型所在设备
                device = next(model.parameters()).device
                context_tensor = context_tensor.to(device)
                
                # 预测
                pred_tensor = model(context_tensor)  # [1, horizon]
                pred = pred_tensor.cpu().numpy().flatten()
                
                # 如果模型输出的 horizon 与期望不一致，进行调整
                model_horizon = model.horizon
                if len(pred) != horizon:
                    print(f"  警告: 模型输出长度 ({len(pred)}) 与期望长度 ({horizon}) 不匹配")
                    if len(pred) > horizon:
                        pred = pred[:horizon]
                    else:
                        # 如果模型输出较短，可能需要重复最后一个值或填充
                        padding = np.zeros(horizon - len(pred))
                        pred = np.concatenate([pred, padding])
        else:
            # 官方 TimesFM 模型
            point_forecast, quantile_forecast = model.forecast(
                horizon=horizon,
                inputs=[context_data]
            )
            
            if isinstance(point_forecast, np.ndarray):
                pred = point_forecast.flatten()
            else:
                pred = np.array(point_forecast).flatten()
        
        # 确保长度正确
        if len(pred) > horizon:
            pred = pred[:horizon]
        elif len(pred) < horizon:
            padding = np.zeros(horizon - len(pred))
            pred = np.concatenate([pred, padding])
        
        return pred
    except Exception as e:
        print(f"预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_results(
    context_data: np.ndarray,
    future_data: np.ndarray,
    predictions: np.ndarray,
    file_name: str,
    save_path: Optional[str] = None
):
    """
    可视化预测结果
    
    Parameters
    ----------
    context_data : np.ndarray
        上下文数据（已标准化）
    future_data : np.ndarray
        真实未来数据（已标准化）
    predictions : np.ndarray
        预测数据（已标准化）
    file_name : str
        文件名
    save_path : str, optional
        保存路径
    """
    plt.figure(figsize=(14, 6))
    
    context_len = len(context_data)
    horizon = len(future_data)
    
    # 绘制上下文数据
    plt.plot(range(context_len), context_data, 'b-', label='Context (历史数据)', linewidth=1.5)
    
    # 绘制真实未来数据
    future_x = range(context_len, context_len + horizon)
    plt.plot(future_x, future_data, 'g-', label='Ground Truth (真实值)', linewidth=1.5)
    
    # 绘制预测数据
    plt.plot(future_x, predictions, 'r--', label='Prediction (预测值)', linewidth=1.5)
    
    # 添加分隔线
    plt.axvline(x=context_len, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value (Normalized)', fontsize=12)
    plt.title(f'TimesFM 预测结果: {file_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TimesFM 推理测试")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/datasets/Pretrain_Data",
        help="数据目录路径"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=720,
        help="上下文长度"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=96,
        help="预测长度"
    )
    parser.add_argument(
        "--num_datasets",
        type=int,
        default=10,
        help="要测试的数据集数量（默认10个）"
    )
    parser.add_argument(
        "--test_multiple_segments",
        action="store_true",
        help="对每个数据集测试多个数据段（滑动窗口）"
    )
    parser.add_argument(
        "--segments_per_dataset",
        type=int,
        default=3,
        help="每个数据集测试的数据段数量（仅在 --test_multiple_segments 时生效）"
    )
    parser.add_argument(
        "--segment_stride",
        type=int,
        default=None,
        help="数据段之间的步长（None表示自动计算）"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/timesfm-2.5-200M-pytorch",
        help="模型ID（HuggingFace Hub 上的模型ID）"
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="本地模型路径（如果提供，将优先使用本地路径而不是 model_id）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理本地模型路径（支持相对路径和绝对路径）
    local_model_path = None
    if args.local_model_path:
        if not os.path.isabs(args.local_model_path):
            # 相对路径：从脚本所在目录开始
            script_dir = Path(__file__).parent
            local_model_path = str(script_dir / args.local_model_path)
        else:
            local_model_path = args.local_model_path
    
    # 加载模型
    model = load_model(args.model_id, args.device, local_path=local_model_path)
    if model is None:
        print("无法加载模型，退出")
        return
    
    # 查找数据文件（支持相对路径和绝对路径）
    if not os.path.isabs(args.data_dir):
        # 相对路径：从脚本所在目录开始
        script_dir = Path(__file__).parent
        data_dir = script_dir / args.data_dir
    else:
        data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        print(f"请检查路径是否正确")
        return
    
    csv_files = list(data_dir.glob("*.csv"))
    if len(csv_files) == 0:
        print(f"在 {data_dir} 中未找到 CSV 文件")
        return
    
    print(f"找到 {len(csv_files)} 个 CSV 文件")
    
    # 尝试加载数据集
    successful_datasets = []
    for csv_file in csv_files:
        if len(successful_datasets) >= args.num_datasets:
            break
        
        print(f"\n尝试加载: {csv_file.name}")
        
        if args.test_multiple_segments:
            # 测试多个数据段
            # 先加载完整数据以确定可用范围
            try:
                df = pd.read_csv(str(csv_file))
                exclude_cols = ['date', 'timestamp', 'time', 'id', 'item_id', 'time_idx']
                numeric_cols = [col for col in df.columns 
                               if col.lower() not in [c.lower() for c in exclude_cols]
                               and pd.api.types.is_numeric_dtype(df[col])]
                
                if len(numeric_cols) == 0:
                    target_col_actual = df.columns[-1]
                else:
                    target_col_actual = numeric_cols[0]
                
                ts_values = df[target_col_actual].values.astype(np.float32)
                
                if len(ts_values) < args.context_length + args.horizon:
                    print(f"✗ 跳过: {csv_file.name} (数据长度不足)")
                    continue
                
                # 计算可用的数据段
                total_length = len(ts_values)
                segment_length = args.context_length + args.horizon
                max_start_idx = total_length - segment_length
                
                # 计算步长
                if args.segment_stride is None:
                    stride = max(1, max_start_idx // args.segments_per_dataset) if max_start_idx > 0 else 1
                else:
                    stride = args.segment_stride
                
                # 生成多个起始索引
                start_indices = []
                for i in range(args.segments_per_dataset):
                    start_idx = min(i * stride, max_start_idx)
                    if start_idx + segment_length <= total_length:
                        start_indices.append(start_idx)
                
                if len(start_indices) == 0:
                    start_indices = [0]  # 至少测试一个段
                
                print(f"  将在 {len(start_indices)} 个数据段上测试: {start_indices}")
                
                # 加载每个数据段
                for seg_idx, start_idx in enumerate(start_indices):
                    result = load_time_series_data(
                        str(csv_file),
                        context_length=args.context_length,
                        horizon=args.horizon,
                        start_idx=start_idx
                    )
                    
                    if result is not None:
                        context_data, future_data, scaler, file_name, original_data, actual_start_idx = result
                        segment_name = f"{file_name}_segment_{seg_idx+1}_start_{actual_start_idx}"
                        successful_datasets.append((context_data, future_data, scaler, segment_name, original_data, actual_start_idx))
                        print(f"  ✓ 成功加载段 {seg_idx+1}/{len(start_indices)}: start_idx={actual_start_idx}")
                    else:
                        print(f"  ✗ 跳过段 {seg_idx+1}/{len(start_indices)}: start_idx={start_idx}")
            except Exception as e:
                print(f"✗ 加载失败: {csv_file.name} - {e}")
                continue
        else:
            # 只测试第一个数据段
            result = load_time_series_data(
                str(csv_file),
                context_length=args.context_length,
                horizon=args.horizon
            )
            
            if result is not None:
                context_data, future_data, scaler, file_name, original_data, start_idx = result
                successful_datasets.append((context_data, future_data, scaler, file_name, original_data, start_idx))
                print(f"✓ 成功加载: {file_name}")
            else:
                print(f"✗ 跳过: {csv_file.name}")
    
    if len(successful_datasets) == 0:
        print("没有成功加载任何数据集")
        return
    
    print(f"\n成功加载 {len(successful_datasets)} 个数据集，开始推理...")
    
    # 对每个数据集进行推理和可视化
    for idx, dataset_info in enumerate(successful_datasets):
        if len(dataset_info) == 6:
            context_data, future_data, scaler, file_name, original_data, start_idx = dataset_info
        else:
            # 兼容旧格式
            context_data, future_data, scaler, file_name, original_data = dataset_info[:5]
            start_idx = 0
        print(f"\n{'='*60}")
        print(f"数据集 {idx + 1}/{len(successful_datasets)}: {file_name}")
        if start_idx > 0:
            print(f"数据段起始索引: {start_idx}")
        print(f"{'='*60}")
        
        # 进行预测
        print("正在进行预测...")
        predictions = predict(model, context_data, horizon=args.horizon)
        
        if predictions is None:
            print("预测失败，跳过该数据集")
            continue
        
        # 打印预测统计信息用于调试
        print(f"  预测结果统计: mean={np.mean(predictions):.4f}, std={np.std(predictions):.4f}, min={np.min(predictions):.4f}, max={np.max(predictions):.4f}")
        
        # 检查预测值是否异常
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            print(f"  警告: 预测结果包含 NaN 或 Inf 值!")
        if np.abs(predictions).max() > 10:
            print(f"  警告: 预测值范围过大 (max abs value: {np.abs(predictions).max():.4f})")
        
        # 计算指标
        mse = np.mean((predictions - future_data) ** 2)
        mae = np.mean(np.abs(predictions - future_data))
        rmse = np.sqrt(mse)
        
        # 计算相关系数
        if np.std(predictions) > 0 and np.std(future_data) > 0:
            correlation = np.corrcoef(predictions, future_data)[0, 1]
        else:
            correlation = 0.0
        
        print(f"预测完成!")
        print(f"  MSE:  {mse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  相关系数: {correlation:.4f}")
        
        # 可视化
        save_path = os.path.join(args.output_dir, f"prediction_{idx+1}_{file_name.replace('.csv', '.png')}")
        visualize_results(
            context_data,
            future_data,
            predictions,
            file_name,
            save_path=save_path
        )
        
        print(f"可视化完成，结果已保存到: {save_path}")
    
    # 汇总所有测试结果
    print(f"\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")
    
    # 收集所有指标
    all_results = []
    for idx, dataset_info in enumerate(successful_datasets):
        if len(dataset_info) == 6:
            context_data, future_data, scaler, file_name, original_data, start_idx = dataset_info
        else:
            context_data, future_data, scaler, file_name, original_data = dataset_info[:5]
            start_idx = 0
        
        # 重新计算指标（如果之前已经计算过）
        predictions = predict(model, context_data, horizon=args.horizon)
        if predictions is not None:
            mse = np.mean((predictions - future_data) ** 2)
            mae = np.mean(np.abs(predictions - future_data))
            rmse = np.sqrt(mse)
            if np.std(predictions) > 0 and np.std(future_data) > 0:
                correlation = np.corrcoef(predictions, future_data)[0, 1]
            else:
                correlation = 0.0
            
            all_results.append({
                'file_name': file_name,
                'start_idx': start_idx,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'correlation': correlation
            })
    
    # 打印汇总表格
    if len(all_results) > 0:
        print(f"\n{'序号':<6} {'数据集名称':<50} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'相关系数':<10}")
        print("-" * 100)
        for idx, result in enumerate(all_results):
            name = result['file_name']
            if len(name) > 47:
                name = name[:44] + "..."
            print(f"{idx+1:<6} {name:<50} {result['mse']:<12.6f} {result['mae']:<12.6f} {result['rmse']:<12.6f} {result['correlation']:<10.4f}")
        
        # 计算平均指标
        avg_mse = np.mean([r['mse'] for r in all_results])
        avg_mae = np.mean([r['mae'] for r in all_results])
        avg_rmse = np.mean([r['rmse'] for r in all_results])
        avg_correlation = np.mean([r['correlation'] for r in all_results])
        
        print("-" * 100)
        print(f"{'平均':<6} {'':<50} {avg_mse:<12.6f} {avg_mae:<12.6f} {avg_rmse:<12.6f} {avg_correlation:<10.4f}")
        
        # 保存汇总结果到文件
        summary_file = os.path.join(args.output_dir, "test_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("TimesFM 测试结果汇总\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"测试时间: {pd.Timestamp.now()}\n")
            f.write(f"模型路径: {local_model_path if local_model_path else args.model_id}\n")
            f.write(f"Context Length: {args.context_length}\n")
            f.write(f"Horizon: {args.horizon}\n")
            f.write(f"测试数据集数量: {len(all_results)}\n\n")
            
            f.write(f"{'序号':<6} {'数据集名称':<50} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'相关系数':<10}\n")
            f.write("-" * 100 + "\n")
            for idx, result in enumerate(all_results):
                f.write(f"{idx+1:<6} {result['file_name']:<50} {result['mse']:<12.6f} {result['mae']:<12.6f} {result['rmse']:<12.6f} {result['correlation']:<10.4f}\n")
            
            f.write("-" * 100 + "\n")
            f.write(f"{'平均':<6} {'':<50} {avg_mse:<12.6f} {avg_mae:<12.6f} {avg_rmse:<12.6f} {avg_correlation:<10.4f}\n")
        
        print(f"\n汇总结果已保存到: {summary_file}")
    
    print(f"\n{'='*60}")
    print("所有测试完成!")
    print(f"结果保存在: {args.output_dir}")
    print(f"共测试了 {len(all_results)} 个数据段")


if __name__ == "__main__":
    main()

