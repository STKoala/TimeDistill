"""
测试学生模型的预测效果
在 ETTh 数据集上测试一条时间序列，并可视化实际值与预测值的对比
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from chronos import Chronos2Pipeline, Chronos2Model
from chronos.chronos2 import Chronos2ForecastingConfig
import warnings
warnings.filterwarnings('ignore')



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
    print(f"正在加载学生模型: {model_path}")
    
    if not os.path.exists(model_path):
        # 尝试查找其他可能的模型路径
        possible_paths = [
            model_path,
            os.path.join(model_path, "pytorch_model.bin"),
            os.path.join(model_path, "model.safetensors"),
            "./chronos-2-distilled/final_model",
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
        print("成功作为 pipeline 加载")
    except Exception as e:
        print(f"无法作为 pipeline 加载，尝试作为模型加载: {e}")
        # 如果只保存了模型，需要创建 pipeline
        try:
            from transformers import AutoConfig
            # 先加载配置
            config = AutoConfig.from_pretrained(model_path)
            # 创建模型
            student_model = Chronos2Model.from_pretrained(model_path)
            # 创建 pipeline
            student_pipeline = Chronos2Pipeline(student_model)
            student_pipeline.model = student_pipeline.model.to(device)
            print("成功作为模型加载并创建 pipeline")
        except Exception as e2:
            print(f"加载失败: {e2}")
            import traceback
            traceback.print_exc()
            raise
    
    print("学生模型加载完成")
    return student_pipeline


def load_teacher_model(model_id: str = "amazon/chronos-2", device: str = "cuda"):
    """
    加载教师模型用于对比
    
    Parameters
    ----------
    model_id : str
        教师模型 ID
    device : str
        设备
        
    Returns
    -------
    Chronos2Pipeline
        教师模型 pipeline
    """
    print(f"正在加载教师模型: {model_id}")
    teacher_pipeline = Chronos2Pipeline.from_pretrained(
        model_id,
        device_map=device
    )
    print("教师模型加载完成")
    return teacher_pipeline


def prepare_test_data(
    data_path: str,
    start_idx: int,
    context_length: int = 96,
    horizon: int = 24,
    target_col: str = "OT"
):
    """
    准备测试数据
    
    Parameters
    ----------
    data_path : str
        数据文件路径
    start_idx : int
        起始索引
    context_length : int
        上下文长度
    horizon : int
        预测长度
    target_col : str
        目标列名
        
    Returns
    -------
    tuple
        (context_data, future_data, context_df)
    """
    df = pd.read_csv(data_path)
    
    # 确定目标列
    if target_col not in df.columns:
        target_col = df.columns[-1]
        print(f"未找到 {target_col} 列，使用最后一列: {target_col}")
    
    # 提取数据
    context_data = df[target_col].iloc[start_idx:start_idx + context_length].values
    future_data = df[target_col].iloc[start_idx + context_length:start_idx + context_length + horizon].values
    
    # 创建 Chronos-2 需要的 DataFrame 格式
    if 'date' in df.columns:
        context_df = pd.DataFrame({
            'id': ['test_series'] * len(context_data),
            'timestamp': pd.to_datetime(df['date'].iloc[start_idx:start_idx + context_length]),
            'target': context_data
        })
    else:
        context_df = pd.DataFrame({
            'id': ['test_series'] * len(context_data),
            'target': context_data
        })
    
    return context_data, future_data, context_df


def prepare_multivariate_test_data(
    data_path: str,
    start_idx: int,
    context_length: int = 96,
    horizon: int = 24,
    target_cols: list = None
):
    """
    准备多变量测试数据
    
    Parameters
    ----------
    data_path : str
        数据文件路径
    start_idx : int
        起始索引
    context_length : int
        上下文长度
    horizon : int
        预测长度
    target_cols : list, optional
        目标列名列表，如果为None则使用所有数值列（排除date等）
        
    Returns
    -------
    tuple
        (context_data_dict, future_data_dict, context_dfs_dict)
        context_data_dict: {col_name: array} 格式的字典
        future_data_dict: {col_name: array} 格式的字典
        context_dfs_dict: {col_name: DataFrame} 格式的字典
    """
    df = pd.read_csv(data_path)
    
    # 确定目标列
    exclude_cols = ['date', 'timestamp', 'time', 'id', 'item_id', 'time_idx']
    if target_cols is None:
        # 使用所有数值列（排除时间列等）
        target_cols = [col for col in df.columns 
                      if col.lower() not in [c.lower() for c in exclude_cols]
                      and pd.api.types.is_numeric_dtype(df[col])]
    
    if len(target_cols) == 0:
        raise ValueError("未找到可用的数值列")
    
    print(f"多变量预测，使用 {len(target_cols)} 个变量: {target_cols}")
    
    context_data_dict = {}
    future_data_dict = {}
    context_dfs_dict = {}
    
    for col in target_cols:
        if col not in df.columns:
            print(f"警告: 列 {col} 不存在，跳过")
            continue
        
        # 提取数据
        context_data = df[col].iloc[start_idx:start_idx + context_length].values
        future_data = df[col].iloc[start_idx + context_length:start_idx + context_length + horizon].values
        
        context_data_dict[col] = context_data
        future_data_dict[col] = future_data
        
        # 创建 Chronos-2 需要的 DataFrame 格式
        if 'date' in df.columns:
            context_df = pd.DataFrame({
                'id': [f'test_series_{col}'] * len(context_data),
                'timestamp': pd.to_datetime(df['date'].iloc[start_idx:start_idx + context_length]),
                'target': context_data
            })
        else:
            context_df = pd.DataFrame({
                'id': [f'test_series_{col}'] * len(context_data),
                'target': context_data
            })
        
        context_dfs_dict[col] = context_df
    
    return context_data_dict, future_data_dict, context_dfs_dict


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
    np.ndarray
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
        print(f"预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_multivariate_with_model(pipeline, context_dfs_dict, horizon: int):
    """
    使用模型进行多变量预测
    
    Parameters
    ----------
    pipeline : Chronos2Pipeline
        模型 pipeline
    context_dfs_dict : dict
        {col_name: DataFrame} 格式的字典
    horizon : int
        预测长度
        
    Returns
    -------
    dict
        {col_name: np.ndarray} 格式的预测结果字典
    """
    predictions_dict = {}
    
    for col_name, context_df in context_dfs_dict.items():
        print(f"  预测变量 {col_name}...")
        try:
            pred = predict_with_model(pipeline, context_df, horizon)
            if pred is not None:
                predictions_dict[col_name] = pred
            else:
                print(f"  警告: 变量 {col_name} 预测失败")
        except Exception as e:
            print(f"  变量 {col_name} 预测出错: {e}")
    
    return predictions_dict


def visualize_predictions(
    context_data: np.ndarray,
    future_data: np.ndarray,
    student_pred: np.ndarray,
    teacher_pred: np.ndarray = None,
    save_path: str = "prediction_comparison.png",
    start_idx: int = None
):
    """
    可视化预测结果
    
    Parameters
    ----------
    context_data : np.ndarray
        上下文数据
    future_data : np.ndarray
        真实未来值
    student_pred : np.ndarray
        学生模型预测
    teacher_pred : np.ndarray, optional
        教师模型预测（用于对比）
    save_path : str
        保存路径
    """
    plt.figure(figsize=(15, 8))
    
    # 时间轴
    context_len = len(context_data)
    horizon_len = len(future_data)
    
    context_time = np.arange(context_len)
    future_time = np.arange(context_len, context_len + horizon_len)
    
    # 绘制上下文数据
    plt.plot(context_time, context_data, 'b-', label='Context', linewidth=2)
    
    # 绘制真实未来值
    plt.plot(future_time, future_data, 'g-', label='Ground Truth', linewidth=2, marker='o', markersize=4)
    
    # 绘制学生模型预测
    if student_pred is not None and len(student_pred) > 0:
        # 确保长度匹配
        min_len = min(len(student_pred), len(future_data))
        plt.plot(
            future_time[:min_len], 
            student_pred[:min_len], 
            'r--', 
            label='Student Model', 
            linewidth=2, 
            marker='s', 
            markersize=4
        )
    
    # 绘制教师模型预测（如果提供）
    if teacher_pred is not None and len(teacher_pred) > 0:
        min_len = min(len(teacher_pred), len(future_data))
        plt.plot(
            future_time[:min_len], 
            teacher_pred[:min_len], 
            'm:', 
            label='Teacher Model', 
            linewidth=2, 
            marker='^', 
            markersize=4
        )
    
    # 添加分隔线
    plt.axvline(x=context_len - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.text(context_len - 0.5, plt.ylim()[1] * 0.95, 'Prediction Start', 
             rotation=90, verticalalignment='top', fontsize=10)
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Time Series Prediction Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 计算并显示指标
    if student_pred is not None and len(student_pred) > 0:
        min_len = min(len(student_pred), len(future_data))
        mse = np.mean((student_pred[:min_len] - future_data[:min_len]) ** 2)
        mae = np.mean(np.abs(student_pred[:min_len] - future_data[:min_len]))
        
        textstr = f'Student Model Metrics:\nMSE: {mse:.4f}\nMAE: {mae:.4f}'
        if teacher_pred is not None and len(teacher_pred) > 0:
            min_len_teacher = min(len(teacher_pred), len(future_data))
            teacher_mse = np.mean((teacher_pred[:min_len_teacher] - future_data[:min_len_teacher]) ** 2)
            teacher_mae = np.mean(np.abs(teacher_pred[:min_len_teacher] - future_data[:min_len_teacher]))
            textstr += f'\n\nTeacher Model Metrics:\nMSE: {teacher_mse:.4f}\nMAE: {teacher_mae:.4f}'
        
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存到: {save_path}")
    plt.show()


def visualize_multivariate_predictions(
    context_data_dict: dict,
    future_data_dict: dict,
    student_pred_dict: dict,
    teacher_pred_dict: dict = None,
    save_dir: str = ".",
    start_idx: int = None
):
    """
    可视化多变量预测结果，每个变量单独生成一张图
    
    Parameters
    ----------
    context_data_dict : dict
        {col_name: np.ndarray} 格式的上下文数据字典
    future_data_dict : dict
        {col_name: np.ndarray} 格式的真实未来值字典
    student_pred_dict : dict
        {col_name: np.ndarray} 格式的学生模型预测字典
    teacher_pred_dict : dict, optional
        {col_name: np.ndarray} 格式的教师模型预测字典（用于对比）
    save_dir : str
        保存目录，图片将保存为 {col_name}_prediction_start{start_idx}.png
    start_idx : int, optional
        起始索引，会添加到文件名中
    """
    n_vars = len(context_data_dict)
    if n_vars == 0:
        print("错误: 没有可用的变量进行可视化")
        return
    
    saved_files = []
    
    # 为每个变量单独生成一张图
    for col_name, context_data in context_data_dict.items():
        future_data = future_data_dict.get(col_name)
        student_pred = student_pred_dict.get(col_name)
        teacher_pred = teacher_pred_dict.get(col_name) if teacher_pred_dict else None
        
        if future_data is None:
            continue
        
        # 时间轴
        context_len = len(context_data)
        horizon_len = len(future_data)
        
        context_time = np.arange(context_len)
        future_time = np.arange(context_len, context_len + horizon_len)
        
        # 创建左右两个子图的figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{col_name} - Prediction Comparison', fontsize=16, fontweight='bold')
        
        # ========== 左半部分：完整视图 (Context + Predict) ==========
        # 绘制上下文数据
        ax1.plot(context_time, context_data, 'b-', label='Context', linewidth=1.5, alpha=0.7)
        
        # 绘制真实未来值
        ax1.plot(future_time, future_data, 'g-', label='Ground Truth', linewidth=2, marker='o', markersize=4)
        
        # 绘制学生模型预测
        if student_pred is not None and len(student_pred) > 0:
            min_len = min(len(student_pred), len(future_data))
            ax1.plot(
                future_time[:min_len], 
                student_pred[:min_len], 
                'r--', 
                label='Student Model', 
                linewidth=2, 
                marker='s', 
                markersize=4
            )
        
        # 绘制教师模型预测（如果提供）
        if teacher_pred is not None and len(teacher_pred) > 0:
            min_len = min(len(teacher_pred), len(future_data))
            ax1.plot(
                future_time[:min_len], 
                teacher_pred[:min_len], 
                'm:', 
                label='Teacher Model', 
                linewidth=2, 
                marker='^', 
                markersize=4
            )
        
        # 添加分隔线
        ax1.axvline(x=context_len - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.text(context_len - 0.5, ax1.get_ylim()[1] * 0.95, 'Prediction Start', 
                 rotation=90, verticalalignment='top', fontsize=9)
        
        ax1.set_xlabel('Time Step', fontsize=11)
        ax1.set_ylabel('Value', fontsize=11)
        ax1.set_title('Full View (Context + Prediction)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # ========== 右半部分：预测部分放大视图 ==========
        # 绘制真实未来值
        ax2.plot(future_time, future_data, 'g-', label='Ground Truth', linewidth=2.5, marker='o', markersize=5)
        
        # 绘制学生模型预测
        if student_pred is not None and len(student_pred) > 0:
            min_len = min(len(student_pred), len(future_data))
            ax2.plot(
                future_time[:min_len], 
                student_pred[:min_len], 
                'r--', 
                label='Student Model', 
                linewidth=2.5, 
                marker='s', 
                markersize=5
            )
        
        # 绘制教师模型预测（如果提供）
        if teacher_pred is not None and len(teacher_pred) > 0:
            min_len = min(len(teacher_pred), len(future_data))
            ax2.plot(
                future_time[:min_len], 
                teacher_pred[:min_len], 
                'm:', 
                label='Teacher Model', 
                linewidth=2.5, 
                marker='^', 
                markersize=5
            )
        
        ax2.set_xlabel('Time Step', fontsize=11)
        ax2.set_ylabel('Value', fontsize=11)
        ax2.set_title('Prediction Zoom View', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 计算并显示指标（显示在右半部分）
        metrics_text = ""
        if student_pred is not None and len(student_pred) > 0:
            min_len = min(len(student_pred), len(future_data))
            mse = np.mean((student_pred[:min_len] - future_data[:min_len]) ** 2)
            mae = np.mean(np.abs(student_pred[:min_len] - future_data[:min_len]))
            rmse = np.sqrt(mse)
            metrics_text = f'Student Model:\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}'
            
            if teacher_pred is not None and len(teacher_pred) > 0:
                min_len_teacher = min(len(teacher_pred), len(future_data))
                teacher_mse = np.mean((teacher_pred[:min_len_teacher] - future_data[:min_len_teacher]) ** 2)
                teacher_mae = np.mean(np.abs(teacher_pred[:min_len_teacher] - future_data[:min_len_teacher]))
                teacher_rmse = np.sqrt(teacher_mse)
                metrics_text += f'\n\nTeacher Model:\nMSE: {teacher_mse:.4f}\nMAE: {teacher_mae:.4f}\nRMSE: {teacher_rmse:.4f}'
        
        if metrics_text:
            ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 保存图片，使用变量名和 start_idx 作为文件名
        if start_idx is not None:
            filename = f'{col_name}_prediction_start{start_idx}.png'
        else:
            filename = f'{col_name}_prediction.png'
        save_path = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存
        saved_files.append(save_path)
        print(f"  {col_name}: 已保存到 {save_path}")
    
    print(f"\n多变量可视化完成，共生成 {len(saved_files)} 张图片")
    return saved_files


def main():
    """主函数"""
    
    # ========== 配置参数 ==========
    # 学生模型路径（如果模型还未训练，可以设置为 None 跳过学生模型测试）
    student_model_path = "./chronos-2-distilled/final_model"  # 修改为你的模型路径
    # 如果模型路径不存在，尝试这些路径
    if not os.path.exists(student_model_path):
        possible_paths = [
            "./chronos-2-distilled/final_model",
            "./chronos-2-distilled",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                student_model_path = path
                break
    
    teacher_model_id = "amazon/chronos-2"  # 或 "amazon/chronos-2-base"
    # data_path = "datasets/eval/ETT-small/ETTm1.csv"
    # data_path = "datasets/eval/traffic/traffic.csv" 
    data_path = "data/datasets/Eval_Data/traffic/traffic.csv"

    # 测试参数
    start_idx = 0  # 从第 500 个时间点开始测试（可以修改）
    context_length = 720
    horizon = 96
    target_col = "OT"
    
    # 从数据路径提取数据集名称（用于创建保存文件夹）
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    # 创建保存目录
    save_base_dir = "results"
    save_dir = os.path.join(save_base_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"结果将保存到: {save_dir}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # ========== 加载模型 ==========
    print("\n" + "=" * 50)
    print("加载模型")
    print("=" * 50)
    
    # 加载学生模型
    student_pipeline = None
    if student_model_path and os.path.exists(student_model_path):
        try:
            student_pipeline = load_student_model(student_model_path, device)
        except Exception as e:
            print(f"无法加载学生模型: {e}")
            print("提示: 将只使用教师模型进行预测")
            student_pipeline = None
    else:
        print(f"学生模型路径不存在: {student_model_path}")
        print("提示: 将只使用教师模型进行预测")
    
    # 加载教师模型（用于对比）
    try:
        teacher_pipeline = load_teacher_model(teacher_model_id, device)
    except Exception as e:
        print(f"无法加载教师模型: {e}")
        teacher_pipeline = None
    
    # ========== 准备测试数据 ==========
    print("\n" + "=" * 50)
    print("准备测试数据")
    print("=" * 50)
    
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        return
    
    context_data, future_data, context_df = prepare_test_data(
        data_path=data_path,
        start_idx=start_idx,
        context_length=context_length,
        horizon=horizon,
        target_col=target_col
    )
    
    print(f"上下文数据长度: {len(context_data)}")
    print(f"未来数据长度: {len(future_data)}")
    print(f"上下文数据范围: [{context_data.min():.2f}, {context_data.max():.2f}]")
    print(f"未来数据范围: [{future_data.min():.2f}, {future_data.max():.2f}]")
    
    # ========== 进行预测 ==========
    print("\n" + "=" * 50)
    print("进行预测")
    print("=" * 50)
    
    # 学生模型预测
    student_pred = None
    if student_pipeline is not None:
        print("学生模型预测中...")
        student_pred = predict_with_model(student_pipeline, context_df, horizon)
        
        if student_pred is None:
            print("学生模型预测失败")
        else:
            print(f"学生模型预测完成，预测值范围: [{student_pred.min():.2f}, {student_pred.max():.2f}]")
    else:
        print("跳过学生模型预测（模型未加载）")
    
    # 教师模型预测（用于对比）
    teacher_pred = None
    if teacher_pipeline is not None:
        print("教师模型预测中...")
        teacher_pred = predict_with_model(teacher_pipeline, context_df, horizon)
        if teacher_pred is not None:
            print(f"教师模型预测完成，预测值范围: [{teacher_pred.min():.2f}, {teacher_pred.max():.2f}]")
    
    # ========== 计算指标 ==========
    print("\n" + "=" * 50)
    print("计算评估指标")
    print("=" * 50)
    
    if student_pred is not None:
        min_len = min(len(student_pred), len(future_data))
        student_mse = np.mean((student_pred[:min_len] - future_data[:min_len]) ** 2)
        student_mae = np.mean(np.abs(student_pred[:min_len] - future_data[:min_len]))
        student_rmse = np.sqrt(student_mse)
        
        print(f"学生模型指标:")
        print(f"  MSE:  {student_mse:.4f}")
        print(f"  MAE:  {student_mae:.4f}")
        print(f"  RMSE: {student_rmse:.4f}")
    
    if teacher_pred is not None:
        min_len = min(len(teacher_pred), len(future_data))
        teacher_mse = np.mean((teacher_pred[:min_len] - future_data[:min_len]) ** 2)
        teacher_mae = np.mean(np.abs(teacher_pred[:min_len] - future_data[:min_len]))
        teacher_rmse = np.sqrt(teacher_mse)
        
        print(f"\n教师模型指标:")
        print(f"  MSE:  {teacher_mse:.4f}")
        print(f"  MAE:  {teacher_mae:.4f}")
        print(f"  RMSE: {teacher_rmse:.4f}")
        
        if student_pred is not None:
            print(f"\n性能对比 (学生/教师):")
            print(f"  MSE 比率:  {student_mse / teacher_mse:.2%}")
            print(f"  MAE 比率:  {student_mae / teacher_mae:.2%}")
            print(f"  RMSE 比率: {student_rmse / teacher_rmse:.2%}")
    
    # ========== 可视化 ==========
    print("\n" + "=" * 50)
    print("生成可视化")
    print("=" * 50)
    
    if student_pred is None and teacher_pred is None:
        print("错误: 没有可用的预测结果进行可视化")
        return
    
    # 构建保存路径，包含 start_idx
    if start_idx is not None:
        save_filename = f"prediction_comparison_start{start_idx}.png"
    else:
        save_filename = "prediction_comparison.png"
    save_path = os.path.join(save_dir, save_filename)
    visualize_predictions(
        context_data=context_data,
        future_data=future_data,
        student_pred=student_pred,
        teacher_pred=teacher_pred,
        save_path=save_path,
        start_idx=start_idx
    )
    
    # ========== 多变量预测 ==========
    print("\n" + "=" * 50)
    print("多变量预测")
    print("=" * 50)
    
    # 准备多变量测试数据
    print("准备多变量测试数据...")
    context_data_dict, future_data_dict, context_dfs_dict = prepare_multivariate_test_data(
        data_path=data_path,
        start_idx=start_idx,
        context_length=context_length,
        horizon=horizon,
        target_cols=None  # None表示使用所有数值列
    )
    
    print(f"多变量数据准备完成，共 {len(context_data_dict)} 个变量")
    for col_name in context_data_dict.keys():
        print(f"  - {col_name}: 上下文长度={len(context_data_dict[col_name])}, 未来长度={len(future_data_dict[col_name])}")
    
    # 多变量预测
    print("\n进行多变量预测...")
    
    # 学生模型多变量预测
    student_pred_dict = {}
    if student_pipeline is not None:
        print("学生模型多变量预测中...")
        student_pred_dict = predict_multivariate_with_model(
            student_pipeline, 
            context_dfs_dict, 
            horizon
        )
        print(f"学生模型多变量预测完成，成功预测 {len(student_pred_dict)}/{len(context_data_dict)} 个变量")
    else:
        print("跳过学生模型多变量预测（模型未加载）")
    
    # 教师模型多变量预测（用于对比）
    teacher_pred_dict = {}
    if teacher_pipeline is not None:
        print("教师模型多变量预测中...")
        teacher_pred_dict = predict_multivariate_with_model(
            teacher_pipeline, 
            context_dfs_dict, 
            horizon
        )
        print(f"教师模型多变量预测完成，成功预测 {len(teacher_pred_dict)}/{len(context_data_dict)} 个变量")
    
    # 计算多变量指标
    print("\n" + "=" * 50)
    print("计算多变量评估指标")
    print("=" * 50)
    
    if student_pred_dict:
        print("\n学生模型多变量指标:")
        for col_name in student_pred_dict.keys():
            if col_name in future_data_dict:
                min_len = min(len(student_pred_dict[col_name]), len(future_data_dict[col_name]))
                mse = np.mean((student_pred_dict[col_name][:min_len] - future_data_dict[col_name][:min_len]) ** 2)
                mae = np.mean(np.abs(student_pred_dict[col_name][:min_len] - future_data_dict[col_name][:min_len]))
                rmse = np.sqrt(mse)
                print(f"  {col_name}: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    if teacher_pred_dict:
        print("\n教师模型多变量指标:")
        for col_name in teacher_pred_dict.keys():
            if col_name in future_data_dict:
                min_len = min(len(teacher_pred_dict[col_name]), len(future_data_dict[col_name]))
                mse = np.mean((teacher_pred_dict[col_name][:min_len] - future_data_dict[col_name][:min_len]) ** 2)
                mae = np.mean(np.abs(teacher_pred_dict[col_name][:min_len] - future_data_dict[col_name][:min_len]))
                rmse = np.sqrt(mse)
                print(f"  {col_name}: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    # 多变量可视化
    print("\n" + "=" * 50)
    print("生成多变量可视化")
    print("=" * 50)
    
    if student_pred_dict or teacher_pred_dict:
        # 每个变量单独生成一张图，保存在数据集文件夹中
        saved_files = visualize_multivariate_predictions(
            context_data_dict=context_data_dict,
            future_data_dict=future_data_dict,
            student_pred_dict=student_pred_dict,
            teacher_pred_dict=teacher_pred_dict,
            save_dir=save_dir,
            start_idx=start_idx
        )
    else:
        print("警告: 没有可用的多变量预测结果进行可视化")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()

