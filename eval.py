"""
评估学生模型在多个数据集上的泛化能力
遍历 dataset/eval/ 目录下的所有数据集，计算评估指标并保存结果
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from chronos import Chronos2Pipeline
import warnings
import json
from datetime import datetime
import logging
import sys
warnings.filterwarnings('ignore')

# 导入数据加载器
# 确保可以导入 data_provider 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from data_provider.data_loader import Dataset_Custom
except ImportError:
    # 如果导入失败，尝试添加父目录到路径
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from TimeDistill.data_provider.data_loader import Dataset_Custom


def setup_logging(log_dir: str = "log", model_name: str = None, 
                  context_length: int = None, horizon: int = None, 
                  dataset_name: str = None):
    """
    设置日志
    
    Parameters
    ----------
    log_dir : str
        日志目录
    model_name : str, optional
        模型名称（用于文件名）
    context_length : int, optional
        上下文长度（用于文件名）
    horizon : int, optional
        预测长度（用于文件名）
    dataset_name : str, optional
        数据集名称（用于文件名）
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 构建文件名组件
    parts = ["eval"]
    
    if model_name:
        # 清理模型名称，移除特殊字符，用于文件名
        safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        safe_model_name = "".join(c for c in safe_model_name if c.isalnum() or c in ("_", "-"))
        parts.append(safe_model_name)
    
    if context_length is not None:
        parts.append(f"ctx{context_length}")
    
    if horizon is not None:
        parts.append(f"pred{horizon}")
    
    if dataset_name:
        # 清理数据集名称，移除特殊字符
        safe_dataset = dataset_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        safe_dataset = "".join(c for c in safe_dataset if c.isalnum() or c in ("_", "-"))
        # 限制长度，避免文件名过长
        if len(safe_dataset) > 30:
            safe_dataset = safe_dataset[:30]
        parts.append(safe_dataset)
    
    # 添加时间戳
    parts.append(datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    # 组合文件名
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


def load_student_model(model_path: str, device: str = "cuda"):
    """
    加载训练好的学生模型
    
    Parameters
    ----------
    model_path : str
        模型路径
    device : str
        设备
        
    Returns
    -------
    Chronos2Pipeline
        加载的模型 pipeline
    """
    logger = logging.getLogger(__name__)
    logger.info(f"正在加载学生模型: {model_path}")
    
    if not os.path.exists(model_path):
        # 尝试查找其他可能的模型路径
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
    
    # 尝试加载模型
    try:
        # 如果保存的是完整的 pipeline
        student_pipeline = Chronos2Pipeline.from_pretrained(
            model_path,
            device_map=device
        )
        logger.info("成功作为 pipeline 加载")
    except Exception as e:
        logger.warning(f"无法作为 pipeline 加载，尝试作为模型加载: {e}")
        # 如果只保存了模型，需要创建 pipeline
        try:
            from transformers import AutoConfig
            # 先加载配置
            config = AutoConfig.from_pretrained(model_path)
            # 创建模型
            from chronos import Chronos2Model
            student_model = Chronos2Model.from_pretrained(model_path)
            # 创建 pipeline
            student_pipeline = Chronos2Pipeline(student_model)
            student_pipeline.model = student_pipeline.model.to(device)
            logger.info("成功作为模型加载并创建 pipeline")
        except Exception as e2:
            logger.error(f"加载失败: {e2}")
            import traceback
            traceback.print_exc()
            raise
    
    logger.info("学生模型加载完成")
    return student_pipeline


def load_teacher_model(model_id: str = "amazon/chronos-2", device: str = "cuda"):
    """
    加载教师模型
    
    Parameters
    ----------
    model_id : str
        教师模型 ID（例如 "amazon/chronos-2" 或 "amazon/chronos-2-base"）
    device : str
        设备
        
    Returns
    -------
    Chronos2Pipeline
        加载的教师模型 pipeline
    """
    logger = logging.getLogger(__name__)
    logger.info(f"正在加载教师模型: {model_id}")
    
    try:
        teacher_pipeline = Chronos2Pipeline.from_pretrained(
            model_id,
            device_map=device
        )
        logger.info("教师模型加载完成")
        return teacher_pipeline
    except Exception as e:
        logger.error(f"教师模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def find_csv_files(eval_dir: str):
    """
    查找 eval 目录下的所有 CSV 文件
    
    Parameters
    ----------
    eval_dir : str
        评估数据集目录
        
    Returns
    -------
    list
        [(dataset_name, csv_path), ...] 格式的列表
    """
    eval_path = Path(eval_dir)
    csv_files = []
    
    if not eval_path.exists():
        logger = logging.getLogger(__name__)
        logger.error(f"评估目录不存在: {eval_dir}")
        return csv_files
    
    # 遍历所有子目录
    for subdir in eval_path.iterdir():
        if subdir.is_dir():
            # 查找该目录下的 CSV 文件
            for csv_file in subdir.glob("*.csv"):
                dataset_name = f"{subdir.name}/{csv_file.stem}"
                csv_files.append((dataset_name, str(csv_file)))
    
    # 也检查 eval_dir 根目录下的 CSV 文件
    for csv_file in eval_path.glob("*.csv"):
        dataset_name = csv_file.stem
        csv_files.append((dataset_name, str(csv_file)))
    
    return csv_files


def find_dataset_file(eval_dir: str, dataset_name: str):
    """
    查找指定的数据集文件
    
    Parameters
    ----------
    eval_dir : str
        评估数据集目录
    dataset_name : str
        数据集名称（可以是 "ETT-small/ETTh1" 或 "ETTh1" 等格式）
        
    Returns
    -------
    tuple or None
        (dataset_name, csv_path) 或 None
    """
    eval_path = Path(eval_dir)
    
    if not eval_path.exists():
        logger = logging.getLogger(__name__)
        logger.error(f"评估目录不存在: {eval_dir}")
        return None
    
    # 尝试直接路径匹配
    # 如果 dataset_name 包含 "/"，尝试作为路径
    if '/' in dataset_name:
        parts = dataset_name.split('/')
        if len(parts) == 2:
            subdir_name, file_name = parts
            csv_path = eval_path / subdir_name / f"{file_name}.csv"
            if csv_path.exists():
                return (dataset_name, str(csv_path))
    
    # 在所有子目录中搜索
    for subdir in eval_path.iterdir():
        if subdir.is_dir():
            # 尝试匹配文件名（不包含扩展名）
            csv_file = subdir / f"{dataset_name}.csv"
            if csv_file.exists():
                return (f"{subdir.name}/{dataset_name}", str(csv_file))
            
            # 也尝试匹配完整路径
            if '/' in dataset_name:
                parts = dataset_name.split('/')
                if len(parts) == 2 and parts[0] == subdir.name:
                    csv_file = subdir / f"{parts[1]}.csv"
                    if csv_file.exists():
                        return (dataset_name, str(csv_file))
    
    # 在根目录搜索
    csv_file = eval_path / f"{dataset_name}.csv"
    if csv_file.exists():
        return (dataset_name, str(csv_file))
    
    return None


def prepare_data(
    data_path: str,
    start_idx: int = None,
    context_length: int = 96,
    horizon: int = 24,
    target_col: str = None
):
    """
    准备测试数据（使用 data_loader.py 的数据加载逻辑）
    
    Parameters
    ----------
    data_path : str
        数据文件路径
    start_idx : int, optional
        起始索引，如果为 None 则从测试集末尾往前取
    context_length : int
        上下文长度
    horizon : int
        预测长度
    target_col : str, optional
        目标列名，如果为 None 则使用最后一列
        
    Returns
    -------
    tuple
        (context_data, future_data, context_df, target_col) 或 None（如果失败）
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 解析路径
        data_path_obj = Path(data_path)
        root_path = str(data_path_obj.parent)
        data_file = data_path_obj.name
        
        # 读取原始数据以确定目标列
        df_raw = pd.read_csv(data_path)
        
        # 确定目标列
        if target_col is None:
            # 排除时间列，使用最后一个数值列
            exclude_cols = ['date', 'timestamp', 'time', 'id', 'item_id', 'time_idx']
            numeric_cols = [col for col in df_raw.columns 
                          if col.lower() not in [c.lower() for c in exclude_cols]
                          and pd.api.types.is_numeric_dtype(df_raw[col])]
            if len(numeric_cols) == 0:
                logger.warning(f"未找到数值列，使用最后一列")
                target_col = df_raw.columns[-1]
            else:
                target_col = numeric_cols[-1]  # 使用最后一个数值列（通常是 OT）
        
        if target_col not in df_raw.columns:
            logger.warning(f"目标列 {target_col} 不存在，使用最后一列")
            target_col = df_raw.columns[-1]
        
        # 使用 Dataset_Custom 加载数据
        # size: [seq_len, label_len, pred_len]
        # 对于评估，label_len 可以设为 0 或与 seq_len 相同
        label_len = context_length  # 使用 context_length 作为 label_len
        dataset = Dataset_Custom(
            root_path=root_path,
            flag='test',  # 使用测试集
            size=[context_length, label_len, horizon],
            features='S',  # 单变量预测
            data_path=data_file,
            target=target_col,
            scale=True,  # 使用归一化
            timeenc=0,  # 使用简单时间编码
            freq='h'  # 默认小时频率，可根据数据调整
        )
        
        # 获取数据集长度
        dataset_len = len(dataset)
        if dataset_len == 0:
            logger.error(f"数据集为空或长度不足")
            return None
        
        # 确定使用的索引
        if start_idx is None:
            # 从测试集末尾往前取
            start_idx = max(0, dataset_len - 1)
        else:
            # 确保索引有效
            start_idx = min(start_idx, dataset_len - 1)
        
        # 获取数据
        seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[start_idx]
        
        # seq_x 是归一化后的 context 数据 (context_length, 1)
        # seq_y 是归一化后的目标数据 (label_len + horizon, 1)
        # 提取 future 部分（最后 horizon 个点）
        context_data_normalized = seq_x.flatten()  # (context_length,)
        future_data_normalized = seq_y[-horizon:].flatten()  # (horizon,)
        
        # 反归一化获取原始值（用于计算指标）
        context_data = dataset.inverse_transform(context_data_normalized.reshape(-1, 1)).flatten()
        future_data = dataset.inverse_transform(future_data_normalized.reshape(-1, 1)).flatten()
        
        # 检查数据有效性
        if len(context_data) < context_length or len(future_data) < horizon:
            logger.warning(f"数据长度不足: context={len(context_data)}, future={len(future_data)}")
            return None
        
        if np.any(np.isnan(context_data)) or np.any(np.isnan(future_data)):
            logger.warning(f"数据包含 NaN 值")
            return None
        
        # 创建 Chronos-2 需要的 DataFrame 格式（使用原始值，不归一化）
        # 需要从原始数据中获取时间戳
        if 'date' in df_raw.columns:
            # 计算在原始数据中的位置
            # Dataset_Custom 的测试集从 border1s[2] 开始
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            # border1s[2] = len(df_raw) - num_test - seq_len
            test_border1 = len(df_raw) - num_test - context_length
            # 在测试集中的实际位置
            actual_start = test_border1 + start_idx
            
            # 确保索引有效
            if actual_start + context_length <= len(df_raw):
                try:
                    timestamps = pd.to_datetime(df_raw['date'].iloc[actual_start:actual_start + context_length])
                    context_df = pd.DataFrame({
                        'id': ['test_series'] * len(context_data),
                        'timestamp': timestamps,
                        'target': context_data
                    })
                except Exception as e:
                    logger.debug(f"时间戳解析失败: {e}")
                    context_df = pd.DataFrame({
                        'id': ['test_series'] * len(context_data),
                        'target': context_data
                    })
            else:
                context_df = pd.DataFrame({
                    'id': ['test_series'] * len(context_data),
                    'target': context_data
                })
        else:
            context_df = pd.DataFrame({
                'id': ['test_series'] * len(context_data),
                'target': context_data
            })
        
        return context_data, future_data, context_df, target_col
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"准备数据失败 {data_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_with_model(pipeline, context_df, horizon: int):
    """
    使用模型进行预测
    
    Parameters
    ----------
    pipeline : Chronos2Pipeline
        模型 pipeline
    context_df : pd.DataFrame
        上下文数据
    horizon : int
        预测长度
        
    Returns
    -------
    np.ndarray or None
        预测值
    """
    try:
        # 使用 predict_df 方法
        pred_df = pipeline.predict_df(
            context_df,
            prediction_length=horizon,
            quantile_levels=[0.5],  # 使用中位数
            id_column="id",
            timestamp_column="timestamp" if "timestamp" in context_df.columns else None,
            target="target",
        )
        
        # 提取预测值
        if '0.5' in pred_df.columns:
            predictions = pred_df['0.5'].values
        elif 'predictions' in pred_df.columns:
            predictions = pred_df['predictions'].values
        elif 'target' in pred_df.columns:
            predictions = pred_df['target'].values
        else:
            # 取第一个数值列
            numeric_cols = [c for c in pred_df.columns 
                          if c not in ['id', 'timestamp', 'item_id', 'time_idx'] 
                          and pd.api.types.is_numeric_dtype(pred_df[c])]
            if len(numeric_cols) > 0:
                predictions = pred_df[numeric_cols[0]].values
            else:
                raise ValueError("无法找到预测值列")
        
        return predictions.flatten()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_metrics(pred: np.ndarray, true: np.ndarray, context_data: np.ndarray = None):
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
        包含 MSE, MAE, RMSE 的字典
    """
    min_len = min(len(pred), len(true))
    if min_len == 0:
        return None
    
    pred = pred[:min_len]
    true = true[:min_len]
    # 
    # if context_data is not None and len(context_data) > 0:
    #     context_mean = np.mean(context_data)
    #     context_std = np.std(context_data)
        
    #     # 避免除零
    #     if context_std == 0:
    #         context_std = 1.0

    #     pred_normalized = (pred - context_mean) / context_std
    #     true_normalized = (true - context_mean) / context_std
        
    #     mse = np.mean((pred_normalized - true_normalized) ** 2)
    #     mae = np.mean(np.abs(pred_normalized - true_normalized))
    # else:
    print(f"没有提供历史信息，使用原始值进行计算")
    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    
    rmse = np.sqrt(mse)
    
    # 计算 MAPE（如果真实值不为0）
    mape = None
    if np.all(true != 0):
        mape = np.mean(np.abs((pred - true) / true)) * 100
    
    return {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape) if mape is not None else None
    }


def evaluate_dataset_sliding_window(
    pipeline,
    dataset_name: str,
    data_path: str,
    context_length: int = 720,
    horizon: int = 96,
    stride: int = 1
):
    """
    使用滑动窗口评估单个数据集的所有窗口（使用 data_loader.py 的数据加载逻辑）
    
    Parameters
    ----------
    pipeline : Chronos2Pipeline
        模型 pipeline
    dataset_name : str
        数据集名称
    data_path : str
        数据文件路径
    context_length : int
        上下文长度
    horizon : int
        预测长度
    stride : int
        滑动窗口步长（默认1，即每次移动1个时间步）
        
    Returns
    -------
    dict or None
        评估结果字典
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"滑动窗口评估数据集: {dataset_name}")
    logger.info(f"数据路径: {data_path}")
    logger.info(f"上下文长度: {context_length}, 预测长度: {horizon}, 步长: {stride}")
    logger.info(f"{'='*60}")
    
    try:
        # 解析路径
        data_path_obj = Path(data_path)
        root_path = str(data_path_obj.parent)
        data_file = data_path_obj.name
        
        # 读取原始数据以确定目标列
        df_raw = pd.read_csv(data_path)
        
        # 确定目标列
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
            features='S',
            data_path=data_file,
            target=target_col,
            scale=True,
            timeenc=0,
            freq='h'
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
        
        logger.info(f"将评估 {total_windows} 个滑动窗口")
        
        # 存储所有预测和真实值
        all_predictions = []
        all_true_values = []
        all_context_data = []  # 存储所有窗口的 context 数据，用于归一化
        successful_windows = 0
        failed_windows = 0
        
        # 计算测试集在原始数据中的起始位置
        # Dataset_Custom 的测试集从 border1s[2] 开始
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        test_border1 = len(df_raw) - num_test - context_length
        
        # 遍历所有窗口
        for window_idx in range(0, dataset_len, stride):
            try:
                # 获取数据
                seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[window_idx]
                
                # 提取 context 和 future 数据（归一化后的）
                context_data_normalized = seq_x.flatten()
                future_data_normalized = seq_y[-horizon:].flatten()
                
                # 反归一化获取原始值
                context_data = dataset.inverse_transform(context_data_normalized.reshape(-1, 1)).flatten()
                future_data = dataset.inverse_transform(future_data_normalized.reshape(-1, 1)).flatten()
                
                # 检查数据有效性
                if len(context_data) < context_length or len(future_data) < horizon:
                    failed_windows += 1
                    continue
                
                if np.any(np.isnan(context_data)) or np.any(np.isnan(future_data)):
                    failed_windows += 1
                    continue
                
                # 创建 Chronos-2 需要的 DataFrame 格式（使用原始值）
                actual_start = test_border1 + window_idx
                if 'date' in df_raw.columns and actual_start + context_length <= len(df_raw):
                    try:
                        timestamps = pd.to_datetime(df_raw['date'].iloc[actual_start:actual_start + context_length])
                        context_df = pd.DataFrame({
                            'id': ['test_series'] * len(context_data),
                            'timestamp': timestamps,
                            'target': context_data
                        })
                    except:
                        context_df = pd.DataFrame({
                            'id': ['test_series'] * len(context_data),
                            'target': context_data
                        })
                else:
                    context_df = pd.DataFrame({
                        'id': ['test_series'] * len(context_data),
                        'target': context_data
                    })
                
                # 进行预测
                predictions = predict_with_model(pipeline, context_df, horizon)
                if predictions is not None and len(predictions) > 0:
                    # 确保长度匹配
                    min_len = min(len(predictions), len(future_data))
                    all_predictions.append(predictions[:min_len])
                    all_true_values.append(future_data[:min_len])
                    all_context_data.append(context_data)  # 保存 context 数据
                    successful_windows += 1
                else:
                    failed_windows += 1
                    
            except Exception as e:
                logger.debug(f"窗口 {window_idx} 处理失败: {e}")
                failed_windows += 1
                continue
            
            # 每100个窗口输出一次进度
            if (window_idx // stride + 1) % 100 == 0:
                logger.info(f"进度: {window_idx // stride + 1}/{total_windows} 窗口 (成功: {successful_windows}, 失败: {failed_windows})")
        
        logger.info(f"窗口评估完成: 成功 {successful_windows}, 失败 {failed_windows}")
        
        if len(all_predictions) == 0:
            logger.error("没有成功的预测窗口")
            return None
        
        # 合并所有预测和真实值
        all_pred = np.concatenate(all_predictions)
        all_true = np.concatenate(all_true_values)
        all_context = np.concatenate(all_context_data)  # 合并所有 context 数据
        
        logger.info(f"总预测点数: {len(all_pred)}")
        logger.info(f"预测值范围: [{all_pred.min():.4f}, {all_pred.max():.4f}]")
        logger.info(f"真实值范围: [{all_true.min():.4f}, {all_true.max():.4f}]")
        
        # 计算整体指标（使用所有 context 数据的均值和方差进行归一化）
        context_mean = np.mean(all_context)
        context_std = np.std(all_context)
        
        # 避免除零
        if context_std == 0:
            context_std = 1.0
        
        # 对预测值和真实值进行归一化
        all_pred_normalized = (all_pred - context_mean) / context_std
        all_true_normalized = (all_true - context_mean) / context_std
        
        # 使用归一化后的值计算指标
        mse = np.mean((all_pred_normalized - all_true_normalized) ** 2)
        mae = np.mean(np.abs(all_pred_normalized - all_true_normalized))
        rmse = np.sqrt(mse)
        
        metrics = {
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': None
        }
        
        # 计算 MAPE（如果真实值不为0，使用原始值）
        if np.all(all_true != 0):
            mape = np.mean(np.abs((all_pred - all_true) / all_true)) * 100
            metrics['MAPE'] = float(mape)
        
        logger.info(f"\n整体评估指标 (基于 {len(all_predictions)} 个窗口):")
        logger.info(f"  MSE:  {metrics['MSE']:.6f}")
        logger.info(f"  MAE:  {metrics['MAE']:.6f}")
        logger.info(f"  RMSE: {metrics['RMSE']:.6f}")
        if metrics['MAPE'] is not None:
            logger.info(f"  MAPE: {metrics['MAPE']:.2f}%")
        
        return {
            'dataset': dataset_name,
            'data_path': data_path,
            'target_col': target_col,
            'context_length': context_length,
            'horizon': horizon,
            'stride': stride,
            'total_windows': total_windows,
            'successful_windows': successful_windows,
            'failed_windows': failed_windows,
            'total_predictions': len(all_pred),
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"滑动窗口评估过程出错 {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_dataset(
    pipeline,
    dataset_name: str,
    data_path: str,
    context_length: int = 96,
    horizon: int = 24,
    start_idx: int = None
):
    """
    评估单个数据集
    
    Parameters
    ----------
    pipeline : Chronos2Pipeline
        模型 pipeline
    dataset_name : str
        数据集名称
    data_path : str
        数据文件路径
    context_length : int
        上下文长度
    horizon : int
        预测长度
    start_idx : int, optional
        起始索引
        
    Returns
    -------
    dict or None
        评估结果字典
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"评估数据集: {dataset_name}")
    logger.info(f"数据路径: {data_path}")
    logger.info(f"{'='*60}")
    
    # 准备数据
    result = prepare_data(
        data_path=data_path,
        start_idx=start_idx,
        context_length=context_length,
        horizon=horizon,
        target_col=None
    )
    
    if result is None:
        logger.error(f"数据准备失败: {dataset_name}")
        return None
    
    context_data, future_data, context_df, target_col = result
    logger.info(f"目标列: {target_col}")
    logger.info(f"上下文长度: {len(context_data)}, 预测长度: {len(future_data)}")
    
    # 进行预测
    try:
        predictions = predict_with_model(pipeline, context_df, horizon)
        if predictions is None:
            logger.error(f"预测失败: {dataset_name}")
            return None
        
        logger.info(f"预测完成，预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        # 计算指标（使用 context_data 进行归一化）
        metrics = calculate_metrics(predictions, future_data, context_data)
        if metrics is None:
            logger.error(f"指标计算失败: {dataset_name}")
            return None
        
        logger.info(f"评估指标:")
        logger.info(f"  MSE:  {metrics['MSE']:.6f}")
        logger.info(f"  MAE:  {metrics['MAE']:.6f}")
        logger.info(f"  RMSE: {metrics['RMSE']:.6f}")
        if metrics['MAPE'] is not None:
            logger.info(f"  MAPE: {metrics['MAPE']:.2f}%")
        
        return {
            'dataset': dataset_name,
            'data_path': data_path,
            'target_col': target_col,
            'context_length': len(context_data),
            'horizon': len(future_data),
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"评估过程出错 {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='评估学生模型或教师模型在多个数据集上的性能')
    parser.add_argument('--model_type', type=str, default='student', choices=['student', 'teacher'],
                        help='模型类型: student (学生模型) 或 teacher (教师模型)')
    parser.add_argument('--model_path', type=str, default='./chronos-2-distilled/final_model',
                        help='学生模型路径（仅在 model_type=student 时使用）')
    parser.add_argument('--teacher_model_id', type=str, default='amazon/chronos-2',
                        help='教师模型 ID（仅在 model_type=teacher 时使用，例如: "amazon/chronos-2" 或 "amazon/chronos-2-base"）')
    parser.add_argument('--eval_dir', type=str, default='datasets/eval',
                        help='评估数据集目录')
    parser.add_argument('--dataset', type=str, default=None,
                        help='指定单个数据集进行评估（使用滑动窗口模式），例如: "ETT-small/ETTh1" 或 "ETTh1"')
    parser.add_argument('--context_length', type=int, default=96,
                        help='上下文长度')
    parser.add_argument('--horizon', type=int, default=24,
                        help='预测长度')
    parser.add_argument('--stride', type=int, default=1,
                        help='滑动窗口步长（仅在 --dataset 模式下有效，默认1）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--log_dir', type=str, default='log',
                        help='日志目录')
    parser.add_argument('--output_file', type=str, default=None,
                        help='结果输出 JSON 文件路径（默认保存在 log 目录）')
    
    args = parser.parse_args()
    
    # 准备模型名称（用于文件名）
    if args.model_type == 'teacher':
        # 从模型ID中提取名称，例如 "amazon/chronos-2" -> "chronos-2"
        model_name = args.teacher_model_id.split("/")[-1] if "/" in args.teacher_model_id else args.teacher_model_id
        model_name = f"teacher_{model_name}"
    else:
        # 从路径中提取模型名称
        model_path = args.model_path
        if os.path.isdir(model_path):
            model_name = os.path.basename(model_path.rstrip("/"))
        else:
            model_name = os.path.basename(os.path.dirname(model_path))
        if not model_name or model_name == ".":
            model_name = "student"
        else:
            model_name = f"student_{model_name}"
    
    # 准备数据集名称（用于文件名）
    dataset_name_for_file = None
    if args.dataset:
        dataset_name_for_file = args.dataset
    
    # 设置日志
    logger, log_file = setup_logging(
        log_dir=args.log_dir,
        model_name=model_name,
        context_length=args.context_length,
        horizon=args.horizon,
        dataset_name=dataset_name_for_file
    )
    
    model_type_name = "教师模型" if args.model_type == 'teacher' else "学生模型"
    logger.info("="*60)
    logger.info(f"开始评估{model_type_name}")
    logger.info("="*60)
    if args.model_type == 'teacher':
        logger.info(f"教师模型 ID: {args.teacher_model_id}")
    else:
        logger.info(f"学生模型路径: {args.model_path}")
    logger.info(f"评估目录: {args.eval_dir}")
    logger.info(f"上下文长度: {args.context_length}")
    logger.info(f"预测长度: {args.horizon}")
    logger.info(f"设备: {args.device}")
    logger.info(f"日志文件: {log_file}")
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，使用 CPU")
        args.device = 'cpu'
    
    # 加载模型
    logger.info("\n" + "="*60)
    logger.info(f"加载{model_type_name}")
    logger.info("="*60)
    try:
        if args.model_type == 'teacher':
            pipeline = load_teacher_model(args.teacher_model_id, args.device)
        else:
            pipeline = load_student_model(args.model_path, args.device)
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return
    
    # 如果指定了单个数据集，使用滑动窗口模式
    if args.dataset is not None:
        logger.info("\n" + "="*60)
        logger.info("单数据集滑动窗口评估模式")
        logger.info("="*60)
        logger.info(f"数据集: {args.dataset}")
        logger.info(f"滑动窗口步长: {args.stride}")
        
        # 查找指定的数据集文件
        dataset_info = find_dataset_file(args.eval_dir, args.dataset)
        if dataset_info is None:
            logger.error(f"未找到数据集: {args.dataset}")
            logger.info("提示: 数据集名称可以是 'ETT-small/ETTh1' 或 'ETTh1' 等格式")
            return
        
        dataset_name, data_path = dataset_info
        logger.info(f"找到数据集: {dataset_name}")
        logger.info(f"数据路径: {data_path}")
        
        # 使用滑动窗口评估
        try:
            result = evaluate_dataset_sliding_window(
                pipeline=pipeline,
                dataset_name=dataset_name,
                data_path=data_path,
                context_length=args.context_length,
                horizon=args.horizon,
                stride=args.stride
            )
            
            if result is not None:
                results = [result]
                successful = 1
                failed = 0
            else:
                results = []
                successful = 0
                failed = 1
        except Exception as e:
            logger.error(f"评估数据集 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            results = []
            successful = 0
            failed = 1
    else:
        # 查找所有数据集
        logger.info("\n" + "="*60)
        logger.info("查找数据集")
        logger.info("="*60)
        csv_files = find_csv_files(args.eval_dir)
        logger.info(f"找到 {len(csv_files)} 个数据集文件")
        for name, path in csv_files:
            logger.info(f"  - {name}: {path}")
        
        if len(csv_files) == 0:
            logger.error("未找到任何数据集文件")
            return
        
        # 评估每个数据集
        results = []
        successful = 0
        failed = 0
        
        for dataset_name, data_path in csv_files:
            try:
                result = evaluate_dataset(
                    pipeline=pipeline,
                    dataset_name=dataset_name,
                    data_path=data_path,
                    context_length=args.context_length,
                    horizon=args.horizon,
                    start_idx=None
                )
                
                if result is not None:
                    results.append(result)
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"评估数据集 {dataset_name} 时出错: {e}")
                failed += 1
    
    # 汇总结果
    logger.info("\n" + "="*60)
    logger.info("评估汇总")
    logger.info("="*60)
    
    if args.dataset is not None:
        logger.info(f"成功: {successful}/1")
        logger.info(f"失败: {failed}/1")
    else:
        total = successful + failed
        logger.info(f"成功: {successful}/{total}")
        logger.info(f"失败: {failed}/{total}")
    
    if len(results) > 0:
        if args.dataset is not None:
            # 单数据集模式，直接显示结果
            result = results[0]
            logger.info("\n最终评估指标:")
            logger.info(f"  MSE:  {result['metrics']['MSE']:.6f}")
            logger.info(f"  MAE:  {result['metrics']['MAE']:.6f}")
            logger.info(f"  RMSE: {result['metrics']['RMSE']:.6f}")
            if result['metrics']['MAPE'] is not None:
                logger.info(f"  MAPE: {result['metrics']['MAPE']:.2f}%")
            logger.info(f"\n窗口统计:")
            logger.info(f"  总窗口数: {result['total_windows']}")
            logger.info(f"  成功窗口: {result['successful_windows']}")
            logger.info(f"  失败窗口: {result['failed_windows']}")
            logger.info(f"  总预测点数: {result['total_predictions']}")
        else:
            # 多数据集模式，计算平均指标
            avg_metrics = {
                'MSE': np.mean([r['metrics']['MSE'] for r in results]),
                'MAE': np.mean([r['metrics']['MAE'] for r in results]),
                'RMSE': np.mean([r['metrics']['RMSE'] for r in results]),
            }
            
            mape_values = [r['metrics']['MAPE'] for r in results if r['metrics']['MAPE'] is not None]
            if len(mape_values) > 0:
                avg_metrics['MAPE'] = np.mean(mape_values)
            
            logger.info("\n平均指标:")
            logger.info(f"  MSE:  {avg_metrics['MSE']:.6f}")
            logger.info(f"  MAE:  {avg_metrics['MAE']:.6f}")
            logger.info(f"  RMSE: {avg_metrics['RMSE']:.6f}")
            if 'MAPE' in avg_metrics:
                logger.info(f"  MAPE: {avg_metrics['MAPE']:.2f}%")
        
        # 保存结果到 JSON
        output_file = args.output_file
        if output_file is None:
            # 构建 JSON 文件名，与日志文件命名保持一致
            parts = ["eval_results"]
            
            # 添加模型名称
            if args.model_type == 'teacher':
                model_name = args.teacher_model_id.split("/")[-1] if "/" in args.teacher_model_id else args.teacher_model_id
                parts.append(f"teacher_{model_name}")
            else:
                model_path = args.model_path
                if os.path.isdir(model_path):
                    model_name = os.path.basename(model_path.rstrip("/"))
                else:
                    model_name = os.path.basename(os.path.dirname(model_path))
                if not model_name or model_name == ".":
                    model_name = "student"
                else:
                    model_name = f"student_{model_name}"
                parts.append(model_name)
            
            # 添加 context 和 horizon
            parts.append(f"ctx{args.context_length}")
            parts.append(f"pred{args.horizon}")
            
            # 添加数据集名称（如果有）
            if args.dataset:
                safe_dataset = args.dataset.replace("/", "_").replace("\\", "_").replace(" ", "_")
                safe_dataset = "".join(c for c in safe_dataset if c.isalnum() or c in ("_", "-"))
                if len(safe_dataset) > 30:
                    safe_dataset = safe_dataset[:30]
                parts.append(safe_dataset)
            else:
                parts.append("multi_dataset")
            
            # 添加时间戳
            parts.append(datetime.now().strftime('%Y%m%d_%H%M%S'))
            
            output_file = os.path.join(args.log_dir, "_".join(parts) + ".json")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_type': args.model_type,
            'eval_dir': args.eval_dir,
            'context_length': args.context_length,
            'horizon': args.horizon,
            'device': args.device,
            'mode': 'sliding_window' if args.dataset else 'multi_dataset',
            'successful': successful,
            'failed': failed,
            'results': results
        }
        
        if args.model_type == 'teacher':
            summary['teacher_model_id'] = args.teacher_model_id
        else:
            summary['model_path'] = args.model_path
        
        if args.dataset:
            summary['dataset'] = args.dataset
            summary['stride'] = args.stride
        else:
            summary['total_datasets'] = successful + failed
            # 计算平均指标
            avg_metrics = {
                'MSE': np.mean([r['metrics']['MSE'] for r in results]),
                'MAE': np.mean([r['metrics']['MAE'] for r in results]),
                'RMSE': np.mean([r['metrics']['RMSE'] for r in results]),
            }
            mape_values = [r['metrics']['MAPE'] for r in results if r['metrics']['MAPE'] is not None]
            if len(mape_values) > 0:
                avg_metrics['MAPE'] = np.mean(mape_values)
            summary['average_metrics'] = avg_metrics
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n结果已保存到: {output_file}")
    
    logger.info("\n" + "="*60)
    logger.info("评估完成")
    logger.info("="*60)


if __name__ == "__main__":
    main()
