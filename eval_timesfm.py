"""
TimesFM 蒸馏模型评估脚本
在测试数据集上评估蒸馏后的学生模型，计算评估指标并与教师模型对比
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import json
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

# 导入训练脚本中的类和函数
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 从训练脚本导入必要的类和函数
from timesfm_distill_gkd import (
    SimpleTimesFMStudent,
    TimesFMDistillationDataset,
    create_student_model
)

# 导入数据加载器
try:
    from data_provider.data_loader import Dataset_Custom
except ImportError:
    # 如果导入失败，尝试添加父目录到路径
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from TimeDistill.data_provider.data_loader import Dataset_Custom
    except ImportError:
        from data.data_provider.data_loader import Dataset_Custom

# 尝试导入 TimesFM（如果可用）
try:
    import timesfm
    TimesFM_2p5_200M_torch = timesfm.TimesFM_2p5_200M_torch
    ForecastConfig = timesfm.ForecastConfig
    TIMESFM_AVAILABLE = True
except (ImportError, AttributeError) as e:
    TIMESFM_AVAILABLE = False
    print(f"警告: timesfm 模块未正确安装或导入失败: {e}")


def setup_logging(log_dir: str = "log", model_name: str = None):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    parts = ["eval_timesfm"]
    if model_name:
        safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        safe_model_name = "".join(c for c in safe_model_name if c.isalnum() or c in ("_", "-"))
        parts.append(safe_model_name)
    
    parts.append(datetime.now().strftime('%Y%m%d_%H%M%S'))
    log_file = os.path.join(log_dir, "_".join(parts) + ".log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), log_file


def load_student_model(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[SimpleTimesFMStudent, Dict[str, Any]]:
    """
    加载训练好的学生模型
    
    Parameters
    ----------
    model_path : str
        模型路径（目录或文件路径）
    device : str
        设备
        
    Returns
    -------
    Tuple[SimpleTimesFMStudent, Dict[str, Any]]
        (模型, 配置信息)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"正在加载学生模型: {model_path}")
    
    # 尝试查找模型文件
    if os.path.isdir(model_path):
        model_file = os.path.join(model_path, "pytorch_model.bin")
    else:
        model_file = model_path
    
    if not os.path.exists(model_file):
        # 尝试其他可能的路径
        possible_paths = [
            model_file,
            os.path.join(model_path, "pytorch_model.bin"),
            "./timesfm-distilled/best_model_epoch_10/pytorch_model.bin",
            "./timesfm-distilled/final_model/pytorch_model.bin",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_file = path
                break
        else:
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
    
    logger.info(f"从 {model_file} 加载模型")
    
    # 加载模型检查点
    checkpoint = torch.load(model_file, map_location=device)
    
    # 获取配置信息
    context_length = checkpoint.get('context_length', 720)
    horizon = checkpoint.get('horizon', 96)
    
    logger.info(f"模型配置: context_length={context_length}, horizon={horizon}")
    
    # 创建学生模型（使用默认配置，因为保存时没有保存完整配置）
    student_model = create_student_model(
        context_length=context_length,
        horizon=horizon
    )
    
    # 加载模型权重
    student_model.load_state_dict(checkpoint['model_state_dict'])
    student_model = student_model.to(device)
    student_model.eval()
    
    logger.info("学生模型加载完成")
    
    config = {
        'context_length': context_length,
        'horizon': horizon
    }
    
    return student_model, config


def load_teacher_model(
    model_id: str = "google/timesfm-2.5-200m-pytorch",
    context_length: int = 720,
    horizon: int = 96,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    加载教师模型用于对比
    
    Parameters
    ----------
    model_id : str
        教师模型 ID
    context_length : int
        上下文长度
    horizon : int
        预测长度
    device : str
        设备
        
    Returns
    -------
    TimesFM 模型
    """
    logger = logging.getLogger(__name__)
    
    if not TIMESFM_AVAILABLE:
        logger.warning("TimesFM 不可用，跳过教师模型加载")
        return None
    
    logger.info(f"正在加载教师模型: {model_id}")
    
    try:
        teacher_model = TimesFM_2p5_200M_torch.from_pretrained(
            model_id,
            torch_compile=True
        )
        
        # 配置模型
        teacher_model.compile(
            ForecastConfig(
                max_context=max(context_length, 1024),
                max_horizon=max(horizon, 256),
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        
        logger.info("教师模型加载完成")
        return teacher_model
    except Exception as e:
        logger.error(f"教师模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    context_data: np.ndarray = None
) -> Dict[str, float]:
    """
    计算评估指标
    
    Parameters
    ----------
    pred : np.ndarray
        预测值
    true : np.ndarray
        真实值
    context_data : np.ndarray, optional
        历史信息（context），用于计算归一化的均值和标准差
        
    Returns
    -------
    dict
        包含 MSE, MAE, RMSE, MAPE 的字典
    """
    min_len = min(len(pred), len(true))
    if min_len == 0:
        return None
    
    pred = pred[:min_len]
    true = true[:min_len]
    
    # 如果提供了 context_data，使用归一化指标
    if context_data is not None and len(context_data) > 0:
        context_mean = np.mean(context_data)
        context_std = np.std(context_data)
        
        if context_std == 0:
            context_std = 1.0
        
        pred_normalized = (pred - context_mean) / context_std
        true_normalized = (true - context_mean) / context_std
        
        mse = np.mean((pred_normalized - true_normalized) ** 2)
        mae = np.mean(np.abs(pred_normalized - true_normalized))
    else:
        mse = np.mean((pred - true) ** 2)
        mae = np.mean(np.abs(pred - true))
    
    rmse = np.sqrt(mse)
    
    # 计算 MAPE（使用原始值）
    mape = None
    if np.all(true != 0):
        mape = np.mean(np.abs((pred - true) / true)) * 100
    
    return {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape) if mape is not None else None
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
    model_type: str = "student"
) -> Dict[str, Any]:
    """
    评估模型在测试集上的表现
    
    Parameters
    ----------
    model : nn.Module
        要评估的模型
    test_loader : DataLoader
        测试数据加载器
    device : str
        设备
    model_type : str
        模型类型（"student" 或 "teacher"）
        
    Returns
    -------
    dict
        评估结果
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始评估 {model_type} 模型...")
    
    # 设置模型为评估模式（如果模型支持）
    if hasattr(model, 'eval'):
        model.eval()
    
    all_predictions = []
    all_targets = []
    all_contexts = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            context = batch["target"].to(device)  # [batch_size, context_length]
            future_target = batch["future_target"].to(device)  # [batch_size, horizon]
            
            try:
                if model_type == "student":
                    # 学生模型直接预测
                    predictions = model(context)  # [batch_size, horizon]
                else:
                    # 教师模型需要特殊处理（参考训练脚本的逻辑）
                    batch_size = context.shape[0]
                    context_np = context.cpu().numpy()
                    
                    # 分批处理以减少内存使用（每批最多 16 个样本）
                    teacher_batch_size = min(16, batch_size)
                    teacher_batch_predictions = []
                    
                    for i in range(0, batch_size, teacher_batch_size):
                        end_idx = min(i + teacher_batch_size, batch_size)
                        batch_inputs = [context_np[j] for j in range(i, end_idx)]
                        
                        try:
                            # 使用 TimesFM 的 forecast 方法
                            point_forecast, quantile_forecast = model.forecast(
                                horizon=future_target.shape[1],
                                inputs=batch_inputs
                            )
                            
                            # 使用点预测（point_forecast）
                            if isinstance(point_forecast, np.ndarray):
                                batch_logits = torch.tensor(point_forecast, dtype=torch.float32)
                            elif torch.is_tensor(point_forecast):
                                batch_logits = point_forecast.float()
                            else:
                                batch_logits = torch.tensor(np.array(point_forecast), dtype=torch.float32)
                            
                            teacher_batch_predictions.append(batch_logits)
                            
                            # 清理 GPU 缓存
                            if device == "cuda":
                                torch.cuda.empty_cache()
                        except Exception as e:
                            logger.warning(f"教师模型预测失败 (批次 [{i}:{end_idx}]): {e}")
                            # 使用零填充作为后备
                            batch_logits = torch.zeros(end_idx - i, future_target.shape[1], dtype=torch.float32)
                            teacher_batch_predictions.append(batch_logits)
                    
                    # 合并所有批次的预测结果
                    if len(teacher_batch_predictions) > 0:
                        predictions = torch.cat(teacher_batch_predictions, dim=0).to(device)
                    else:
                        # 如果所有批次都失败，使用零填充
                        predictions = torch.zeros(batch_size, future_target.shape[1], dtype=torch.float32).to(device)
                
                # 确保形状匹配
                min_len = min(predictions.shape[1], future_target.shape[1])
                predictions = predictions[:, :min_len]
                future_target = future_target[:, :min_len]
                
                # 收集结果
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(future_target.cpu().numpy())
                all_contexts.append(context.cpu().numpy())
                
            except Exception as e:
                logger.warning(f"批次 {batch_idx} 评估失败: {e}")
                continue
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  已处理 {batch_idx + 1}/{len(test_loader)} 个批次")
    
    if len(all_predictions) == 0:
        logger.error("没有成功的预测")
        return None
    
    # 合并所有结果
    all_pred = np.concatenate(all_predictions, axis=0)
    all_true = np.concatenate(all_targets, axis=0)
    all_ctx = np.concatenate(all_contexts, axis=0)
    
    # 展平为 1D 数组
    all_pred_flat = all_pred.flatten()
    all_true_flat = all_true.flatten()
    all_ctx_flat = all_ctx.flatten()
    
    logger.info(f"总预测点数: {len(all_pred_flat)}")
    logger.info(f"预测值范围: [{all_pred_flat.min():.4f}, {all_pred_flat.max():.4f}]")
    logger.info(f"真实值范围: [{all_true_flat.min():.4f}, {all_true_flat.max():.4f}]")
    
    # 计算指标
    metrics = calculate_metrics(all_pred_flat, all_true_flat, all_ctx_flat)
    
    return {
        'metrics': metrics,
        'predictions': all_pred_flat,
        'targets': all_true_flat,
        'contexts': all_ctx_flat
    }


def evaluate_with_dataset_custom(
    model: nn.Module,
    data_path: str,
    context_length: int,
    horizon: int,
    device: str = "cuda",
    model_type: str = "student",
    stride: int = 1,
    target_col: str = None,
    freq: str = 'h'
) -> Dict[str, Any]:
    """
    使用 Dataset_Custom 进行滑动窗口评估
    
    Parameters
    ----------
    model : nn.Module
        要评估的模型
    data_path : str
        数据文件路径
    context_length : int
        上下文长度
    horizon : int
        预测长度
    device : str
        设备
    model_type : str
        模型类型（"student" 或 "teacher"）
    stride : int
        滑动窗口步长（默认1）
    target_col : str, optional
        目标列名
    freq : str
        时间频率（默认 'h'）
        
    Returns
    -------
    dict
        评估结果
    """
    logger = logging.getLogger(__name__)
    logger.info(f"使用 Dataset_Custom 评估数据集: {os.path.basename(data_path)}")
    
    # 解析路径
    data_path_obj = Path(data_path)
    root_path = str(data_path_obj.parent)
    data_file = data_path_obj.name
    
    # 读取原始数据以确定目标列
    df_raw = pd.read_csv(data_path)
    
    # 确定目标列
    if target_col is None:
        exclude_cols = ['date', 'timestamp', 'time', 'id', 'item_id', 'time_idx']
        numeric_cols = [col for col in df_raw.columns 
                      if col.lower() not in [c.lower() for c in exclude_cols]
                      and pd.api.types.is_numeric_dtype(df_raw[col])]
        if len(numeric_cols) == 0:
            logger.warning(f"未找到数值列，使用最后一列")
            target_col = df_raw.columns[-1]
        else:
            target_col = numeric_cols[-1]
    
    if target_col not in df_raw.columns:
        logger.warning(f"目标列 {target_col} 不存在，使用最后一列")
        target_col = df_raw.columns[-1]
    
    logger.info(f"目标列: {target_col}")
    logger.info(f"数据总长度: {len(df_raw)}")
    
    # 使用 Dataset_Custom 加载数据
    label_len = context_length
    dataset = Dataset_Custom(
        root_path=root_path,
        flag='test',  # 使用测试集
        size=[context_length, label_len, horizon],
        features='S',  # 单变量预测
        data_path=data_file,
        target=target_col,
        scale=True,  # 使用归一化
        timeenc=0,  # 使用简单时间编码
        freq=freq
    )
    
    dataset_len = len(dataset)
    if dataset_len == 0:
        logger.error(f"数据集为空或长度不足")
        return None
    
    # 计算所有可能的窗口（考虑 stride）
    total_windows = (dataset_len - 1) // stride + 1
    if total_windows <= 0:
        logger.error(f"数据长度不足，无法创建滑动窗口")
        return None
    
    logger.info(f"将评估 {total_windows} 个滑动窗口（步长: {stride}）")
    
    # 设置模型为评估模式（如果模型支持）
    if hasattr(model, 'eval'):
        model.eval()
    
    all_predictions = []
    all_targets = []
    all_contexts = []
    successful_windows = 0
    failed_windows = 0
    
    with torch.no_grad():
        for window_idx in range(0, dataset_len, stride):
            try:
                # 获取数据
                seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[window_idx]
                
                # seq_x 是归一化后的 context 数据 (context_length, 1)
                # seq_y 是归一化后的目标数据 (label_len + horizon, 1)
                # 提取 future 部分（最后 horizon 个点）
                context_normalized = seq_x.flatten()  # (context_length,)
                future_normalized = seq_y[-horizon:].flatten()  # (horizon,)
                
                # 转换为 tensor
                context_tensor = torch.tensor(context_normalized, dtype=torch.float32).unsqueeze(0).to(device)  # [1, context_length]
                
                # 进行预测
                if model_type == "student":
                    # 学生模型直接预测
                    pred_normalized = model(context_tensor).squeeze(0).cpu().numpy()  # (horizon,)
                else:
                    # 教师模型需要特殊处理
                    context_np = context_normalized.reshape(-1)
                    try:
                        point_forecast, _ = model.forecast(
                            horizon=horizon,
                            inputs=[context_np]
                        )
                        if isinstance(point_forecast, np.ndarray):
                            pred_normalized = point_forecast.flatten()
                        else:
                            pred_normalized = np.array(point_forecast).flatten()
                        
                        # 确保长度正确
                        if len(pred_normalized) > horizon:
                            pred_normalized = pred_normalized[:horizon]
                        elif len(pred_normalized) < horizon:
                            padding = np.zeros(horizon - len(pred_normalized))
                            pred_normalized = np.concatenate([pred_normalized, padding])
                    except Exception as e:
                        logger.warning(f"教师模型预测失败 (窗口 {window_idx}): {e}")
                        pred_normalized = np.zeros(horizon)
                
                # 反归一化获取原始值
                context_original = dataset.inverse_transform(context_normalized.reshape(-1, 1)).flatten()
                future_original = dataset.inverse_transform(future_normalized.reshape(-1, 1)).flatten()
                pred_original = dataset.inverse_transform(pred_normalized.reshape(-1, 1)).flatten()
                
                # 收集结果
                all_predictions.append(pred_original)
                all_targets.append(future_original)
                all_contexts.append(context_original)
                
                successful_windows += 1
                
            except Exception as e:
                logger.warning(f"窗口 {window_idx} 评估失败: {e}")
                failed_windows += 1
                continue
            
            if (window_idx // stride + 1) % 100 == 0:
                logger.info(f"进度: {window_idx // stride + 1}/{total_windows} 窗口 (成功: {successful_windows}, 失败: {failed_windows})")
    
    logger.info(f"窗口评估完成: 成功 {successful_windows}, 失败 {failed_windows}")
    
    if len(all_predictions) == 0:
        logger.error("没有成功的预测窗口")
        return None
    
    # 合并所有预测和真实值
    all_pred = np.concatenate(all_predictions)
    all_true = np.concatenate(all_targets)
    all_ctx = np.concatenate(all_contexts)
    
    logger.info(f"总预测点数: {len(all_pred)}")
    logger.info(f"预测值范围: [{all_pred.min():.4f}, {all_pred.max():.4f}]")
    logger.info(f"真实值范围: [{all_true.min():.4f}, {all_true.max():.4f}]")
    
    # 计算指标（使用 context 数据进行归一化）
    metrics = calculate_metrics(all_pred, all_true, all_ctx)
    
    return {
        'metrics': metrics,
        'predictions': all_pred,
        'targets': all_true,
        'contexts': all_ctx,
        'successful_windows': successful_windows,
        'failed_windows': failed_windows,
        'total_windows': total_windows
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="评估 TimesFM 蒸馏模型")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./timesfm-distilled/best_model_epoch_10",
        help="学生模型路径"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="测试数据目录"
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
        "--batch_size",
        type=int,
        default=32,
        help="批次大小"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备"
    )
    parser.add_argument(
        "--compare_teacher",
        action="store_true",
        help="是否与教师模型对比"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=20,
        help="最大处理文件数"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="每个文件最大样本数"
    )
    parser.add_argument(
        "--use_dataset_custom",
        action="store_true",
        help="使用 Dataset_Custom 进行评估（使用 data_loader.py 的数据处理方法）"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="数据集名称（如 ETT-small），如果指定则只评估该数据集目录下的文件"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="滑动窗口步长（默认1，即每个窗口都评估）"
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default=None,
        help="目标列名（如果为 None 则自动检测）"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default='h',
        help="时间频率（默认 'h'，对于 ETTm 系列使用 't'）"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger, log_file = setup_logging(model_name=os.path.basename(args.model_path))
    logger.info("=" * 60)
    logger.info("TimesFM 蒸馏模型评估")
    logger.info("=" * 60)
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"设备: {args.device}")
    logger.info(f"日志文件: {log_file}")
    
    # 加载学生模型
    try:
        student_model, config = load_student_model(args.model_path, args.device)
        context_length = config['context_length']
        horizon = config['horizon']
        logger.info(f"使用模型配置: context_length={context_length}, horizon={horizon}")
    except Exception as e:
        logger.error(f"加载学生模型失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 如果使用 Dataset_Custom 模式
    if args.use_dataset_custom:
        # 准备数据目录
        if args.data_dir is None:
            # 尝试 Eval_Data 目录
            possible_data_dirs = [
                "data/datasets/Eval_Data",
                "./data/datasets/Eval_Data",
                "../data/datasets/Eval_Data",
            ]
            for data_dir in possible_data_dirs:
                if os.path.exists(data_dir) and os.path.isdir(data_dir):
                    args.data_dir = data_dir
                    logger.info(f"找到数据目录: {args.data_dir}")
                    break
        
        if args.data_dir is None:
            logger.error("未找到数据目录，请使用 --data_dir 指定")
            return
        
        # 如果指定了数据集名称，只评估该数据集目录
        if args.dataset_name:
            eval_dir = os.path.join(args.data_dir, args.dataset_name)
            if not os.path.exists(eval_dir):
                logger.error(f"数据集目录不存在: {eval_dir}")
                return
            data_files = list(Path(eval_dir).glob("*.csv"))
        else:
            # 评估所有数据集目录
            eval_dir = args.data_dir
            data_files = []
            for subdir in os.listdir(eval_dir):
                subdir_path = os.path.join(eval_dir, subdir)
                if os.path.isdir(subdir_path):
                    data_files.extend(list(Path(subdir_path).glob("*.csv")))
        
        if len(data_files) == 0:
            logger.error("未找到任何 CSV 文件")
            return
        
        logger.info(f"找到 {len(data_files)} 个数据文件")
        
        # 评估每个文件
        all_results = []
        for data_file in data_files:
            data_path = str(data_file)
            dataset_name = data_file.stem
            
            logger.info("\n" + "=" * 60)
            logger.info(f"评估数据集: {dataset_name}")
            logger.info(f"文件路径: {data_path}")
            logger.info("=" * 60)
            
            # 根据文件名确定频率
            freq = args.freq
            if 'ETTm' in dataset_name:
                freq = 't'  # 分钟级数据
            
            # 评估学生模型
            student_results = evaluate_with_dataset_custom(
                model=student_model,
                data_path=data_path,
                context_length=context_length,
                horizon=horizon,
                device=args.device,
                model_type="student",
                stride=args.stride,
                target_col=args.target_col,
                freq=freq
            )
            
            if student_results is None:
                logger.warning(f"数据集 {dataset_name} 评估失败，跳过")
                continue
            
            logger.info(f"\n{dataset_name} - 学生模型评估指标:")
            logger.info(f"  成功窗口: {student_results['successful_windows']}/{student_results['total_windows']}")
            logger.info(f"  MSE:  {student_results['metrics']['MSE']:.6f}")
            logger.info(f"  MAE:  {student_results['metrics']['MAE']:.6f}")
            logger.info(f"  RMSE: {student_results['metrics']['RMSE']:.6f}")
            if student_results['metrics']['MAPE'] is not None:
                logger.info(f"  MAPE: {student_results['metrics']['MAPE']:.2f}%")
            
            # 评估教师模型（如果启用）
            teacher_results = None
            if args.compare_teacher:
                teacher_model = load_teacher_model(
                    context_length=context_length,
                    horizon=horizon,
                    device=args.device
                )
                
                if teacher_model is not None:
                    teacher_results = evaluate_with_dataset_custom(
                        model=teacher_model,
                        data_path=data_path,
                        context_length=context_length,
                        horizon=horizon,
                        device=args.device,
                        model_type="teacher",
                        stride=args.stride,
                        target_col=args.target_col,
                        freq=freq
                    )
                    
                    if teacher_results is not None:
                        logger.info(f"\n{dataset_name} - 教师模型评估指标:")
                        logger.info(f"  MSE:  {teacher_results['metrics']['MSE']:.6f}")
                        logger.info(f"  MAE:  {teacher_results['metrics']['MAE']:.6f}")
                        logger.info(f"  RMSE: {teacher_results['metrics']['RMSE']:.6f}")
                        if teacher_results['metrics']['MAPE'] is not None:
                            logger.info(f"  MAPE: {teacher_results['metrics']['MAPE']:.2f}%")
                        
                        # 对比结果
                        logger.info(f"\n{dataset_name} - 模型对比:")
                        logger.info(f"  MSE 比率 (学生/教师): {student_results['metrics']['MSE'] / teacher_results['metrics']['MSE']:.4f}")
                        logger.info(f"  MAE 比率 (学生/教师): {student_results['metrics']['MAE'] / teacher_results['metrics']['MAE']:.4f}")
                        logger.info(f"  RMSE 比率 (学生/教师): {student_results['metrics']['RMSE'] / teacher_results['metrics']['RMSE']:.4f}")
            
            # 保存结果
            result = {
                'dataset': dataset_name,
                'data_path': data_path,
                'student_metrics': student_results['metrics'],
                'successful_windows': student_results['successful_windows'],
                'total_windows': student_results['total_windows']
            }
            
            if teacher_results is not None:
                result['teacher_metrics'] = teacher_results['metrics']
                result['comparison'] = {
                    'mse_ratio': student_results['metrics']['MSE'] / teacher_results['metrics']['MSE'],
                    'mae_ratio': student_results['metrics']['MAE'] / teacher_results['metrics']['MAE'],
                    'rmse_ratio': student_results['metrics']['RMSE'] / teacher_results['metrics']['RMSE']
                }
            
            all_results.append(result)
        
        # 保存所有结果
        results = {
            'model_path': args.model_path,
            'context_length': context_length,
            'horizon': horizon,
            'stride': args.stride,
            'data_dir': args.data_dir,
            'dataset_name': args.dataset_name,
            'results': all_results
        }
        
        results_file = log_file.replace('.log', '_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n所有结果已保存到: {results_file}")
        logger.info("=" * 60)
        logger.info("评估完成！")
        logger.info("=" * 60)
        return
    
    # 原有的评估模式（使用 TimesFMDistillationDataset）
    # 准备测试数据
    if args.data_dir is None:
        # 尝试多个可能的数据路径
        possible_data_dirs = [
            "data/datasets/Pretrain_Data",
            "datasets/monash_csv_downsmp",
            "data/datasets/monash_csv_downsmp",
            "./datasets/monash_csv_downsmp",
        ]
        for data_dir in possible_data_dirs:
            if os.path.exists(data_dir) and os.path.isdir(data_dir):
                args.data_dir = data_dir
                logger.info(f"找到数据目录: {args.data_dir}")
                break
        
        if args.data_dir is None:
            logger.error("未找到数据目录，请使用 --data_dir 指定")
            return
    
    logger.info("=" * 60)
    logger.info("准备测试数据...")
    logger.info("=" * 60)
    
    test_dataset = TimesFMDistillationDataset(
        data_paths=[],
        data_dir=args.data_dir,
        dataset_names=None,
        context_length=context_length,
        horizon=horizon,
        stride=horizon,  # 测试时使用非重叠窗口
        target_col=None,
        scale=True,
        split='test',
        train_split=0.8,
        test_split=0.1,
        max_files=args.max_files,
        max_samples_per_file=args.max_samples
    )
    
    if len(test_dataset) == 0:
        logger.error("测试数据集为空！")
        return
    
    logger.info(f"测试样本数: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 评估学生模型
    logger.info("=" * 60)
    logger.info("评估学生模型...")
    logger.info("=" * 60)
    
    student_results = evaluate_model(
        student_model,
        test_loader,
        args.device,
        model_type="student"
    )
    
    if student_results is None:
        logger.error("学生模型评估失败")
        return
    
    logger.info("\n学生模型评估指标:")
    logger.info(f"  MSE:  {student_results['metrics']['MSE']:.6f}")
    logger.info(f"  MAE:  {student_results['metrics']['MAE']:.6f}")
    logger.info(f"  RMSE: {student_results['metrics']['RMSE']:.6f}")
    if student_results['metrics']['MAPE'] is not None:
        logger.info(f"  MAPE: {student_results['metrics']['MAPE']:.2f}%")
    
    # 评估教师模型（如果启用）
    teacher_results = None
    if args.compare_teacher:
        logger.info("\n" + "=" * 60)
        logger.info("加载并评估教师模型...")
        logger.info("=" * 60)
        
        teacher_model = load_teacher_model(
            context_length=context_length,
            horizon=horizon,
            device=args.device
        )
        
        if teacher_model is not None:
            teacher_results = evaluate_model(
                teacher_model,
                test_loader,
                args.device,
                model_type="teacher"
            )
            
            if teacher_results is not None:
                logger.info("\n教师模型评估指标:")
                logger.info(f"  MSE:  {teacher_results['metrics']['MSE']:.6f}")
                logger.info(f"  MAE:  {teacher_results['metrics']['MAE']:.6f}")
                logger.info(f"  RMSE: {teacher_results['metrics']['RMSE']:.6f}")
                if teacher_results['metrics']['MAPE'] is not None:
                    logger.info(f"  MAPE: {teacher_results['metrics']['MAPE']:.2f}%")
                
                # 对比结果
                logger.info("\n" + "=" * 60)
                logger.info("模型对比:")
                logger.info("=" * 60)
                logger.info(f"MSE 比率 (学生/教师): {student_results['metrics']['MSE'] / teacher_results['metrics']['MSE']:.4f}")
                logger.info(f"MAE 比率 (学生/教师): {student_results['metrics']['MAE'] / teacher_results['metrics']['MAE']:.4f}")
                logger.info(f"RMSE 比率 (学生/教师): {student_results['metrics']['RMSE'] / teacher_results['metrics']['RMSE']:.4f}")
    
    # 保存结果
    results = {
        'student_metrics': student_results['metrics'],
        'model_path': args.model_path,
        'context_length': context_length,
        'horizon': horizon,
        'test_samples': len(test_dataset),
        'data_dir': args.data_dir
    }
    
    if teacher_results is not None:
        results['teacher_metrics'] = teacher_results['metrics']
        results['comparison'] = {
            'mse_ratio': student_results['metrics']['MSE'] / teacher_results['metrics']['MSE'],
            'mae_ratio': student_results['metrics']['MAE'] / teacher_results['metrics']['MAE'],
            'rmse_ratio': student_results['metrics']['RMSE'] / teacher_results['metrics']['RMSE']
        }
    
    results_file = log_file.replace('.log', '_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n结果已保存到: {results_file}")
    logger.info("=" * 60)
    logger.info("评估完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

