"""
设备管理工具函数
"""

import torch
import torch.nn as nn


def move_model_to_device(model: nn.Module, device: str) -> nn.Module:
    """
    将模型移动到指定设备
    
    注意：如果模型使用了accelerate hooks，不能直接移动。
    在这种情况下，应该使用accelerate的API来管理设备，或者禁用accelerate。
    
    Parameters
    ----------
    model : nn.Module
        要移动的模型
    device : str
        目标设备（如 "cuda:0" 或 "cpu"）
    
    Returns
    -------
    nn.Module
        移动后的模型
    """
    # 检查模型是否使用了accelerate hooks
    # 如果使用了accelerate，直接调用.to()会触发警告
    # 但在我们的场景中，我们在加载时已经设置了device_map=None
    # 所以应该可以安全地移动
    
    try:
        model = model.to(device)
    except RuntimeError as e:
        if "accelerate hooks" in str(e).lower() or "dispatched" in str(e).lower():
            raise RuntimeError(
                f"模型使用了accelerate hooks，无法直接移动。"
                f"请确保在加载模型时设置device_map=None以禁用accelerate的自动设备映射。"
            ) from e
        raise
    
    return model

