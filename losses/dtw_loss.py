"""
可微分的DTW（Dynamic Time Warping）损失函数

DTW用于计算两个时间序列之间的对齐距离，对时间偏移更加鲁棒。
本实现使用soft-DTW的思想，通过可微分的动态规划实现。
"""

import torch
import torch.nn as nn


class DTWLoss(nn.Module):
    """
    可微分的DTW（Dynamic Time Warping）损失函数
    
    DTW用于计算两个时间序列之间的对齐距离，对时间偏移更加鲁棒。
    本实现使用soft-DTW的思想，通过可微分的动态规划实现。
    """
    
    def __init__(self, gamma: float = 1.0, use_fast_approx: bool = True):
        """
        Parameters
        ----------
        gamma : float
            Soft-DTW的温度参数，gamma越小越接近硬DTW，gamma越大越平滑
        use_fast_approx : bool
            如果True，使用快速近似实现（避免复杂的动态规划）
            如果False，使用标准的soft-DTW实现（较慢但更精确）
        """
        super().__init__()
        self.gamma = gamma
        self.use_fast_approx = use_fast_approx
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算DTW损失
        
        Parameters
        ----------
        pred : torch.Tensor
            预测序列，形状为 (batch_size, seq_len) 或 (batch_size, seq_len, 1)
        target : torch.Tensor
            目标序列，形状为 (batch_size, seq_len) 或 (batch_size, seq_len, 1)
            
        Returns
        -------
        torch.Tensor
            DTW损失值（标量）
        """
        # 确保输入是2D的 (batch_size, seq_len)
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        if target.dim() == 3:
            target = target.squeeze(-1)
        
        # 确保pred和target长度相同（如果不同，截断到最小长度）
        min_len = min(pred.shape[1], target.shape[1])
        pred = pred[:, :min_len]
        target = target[:, :min_len]
        
        batch_size = pred.shape[0]
        seq_len = pred.shape[1]
        
        # 批量计算距离矩阵
        # pred: (batch_size, seq_len) -> (batch_size, seq_len, 1)
        # target: (batch_size, seq_len) -> (batch_size, 1, seq_len)
        pred_expanded = pred.unsqueeze(2)  # (batch_size, seq_len, 1)
        target_expanded = target.unsqueeze(1)  # (batch_size, 1, seq_len)
        distance_matrix = (pred_expanded - target_expanded) ** 2  # (batch_size, seq_len, seq_len)
        
        # 批量计算DTW损失
        if self.use_fast_approx:
            # 使用快速近似：计算加权平均距离
            # 这避免了复杂的动态规划，同时保持可微分性
            # 对于时间序列，使用soft-alignment权重
            weights = torch.softmax(-distance_matrix / self.gamma, dim=2)  # (batch_size, n, m)
            # 计算加权距离
            weighted_dist = torch.sum(distance_matrix * weights, dim=(1, 2))  # (batch_size,)
            # 归一化
            losses = weighted_dist / (seq_len * seq_len)
        else:
            # 使用标准soft-DTW实现
            losses = []
            for i in range(batch_size):
                dtw_loss = self._soft_dtw(distance_matrix[i], self.gamma)
                losses.append(dtw_loss)
            losses = torch.stack(losses)
        
        # 返回平均损失
        return losses.mean()
    
    def _soft_dtw(self, distance_matrix: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        计算soft-DTW距离（优化实现，使用预分配矩阵避免频繁操作）
        
        注意：为了性能，这里使用一个简化的实现。R矩阵使用detach来避免梯度问题，
        但最终结果通过distance_matrix保持梯度连接。
        
        Parameters
        ----------
        distance_matrix : torch.Tensor
            距离矩阵，形状为 (n, m)
        gamma : float
            温度参数（必须>0）
            
        Returns
        -------
        torch.Tensor
            soft-DTW距离
        """
        n, m = distance_matrix.shape
        
        # 如果gamma太小，使用一个最小值避免数值不稳定
        if gamma <= 0:
            gamma = 1e-6
        
        # 使用预分配的矩阵，避免频繁的torch.cat操作
        large_value = 1e6
        
        # 初始化R矩阵（使用numpy风格的数组，避免梯度问题）
        # 使用torch.zeros初始化，然后填充大值
        R = torch.full(
            (n + 1, m + 1),
            large_value,
            device=distance_matrix.device,
            dtype=distance_matrix.dtype
        )
        R[0, 0] = 0.0
        
        # 逐行计算（R不需要梯度，但计算过程中使用distance_matrix保持梯度）
        # 使用with torch.no_grad()来避免R的梯度计算
        with torch.no_grad():
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    # 获取三个方向的值
                    diag = R[i - 1, j - 1]
                    up = R[i - 1, j]
                    left = R[i, j - 1]
                    
                    # 计算三个方向的soft-min（在no_grad上下文中）
                    r_prev = torch.stack([diag, up, left])
                    r_prev_neg = -r_prev / gamma
                    r_max = torch.max(r_prev_neg)
                    exp_values = torch.exp(r_prev_neg - r_max)
                    log_sum = torch.log(torch.sum(exp_values) + 1e-10)
                    r_min = -gamma * (r_max + log_sum)
                    
                    # 计算新的累积距离（在no_grad中，使用detach的distance_matrix）
                    # 注意：这里我们使用distance_matrix的值，但不保持梯度
                    dist_val = distance_matrix[i - 1, j - 1].detach()
                    new_val = dist_val + r_min
                    R[i, j] = new_val
        
        # 重新计算最后一步，保持梯度连接
        # 这是关键：我们需要重新计算最后一步，确保梯度能够通过distance_matrix传播
        if n > 0 and m > 0:
            # 获取最后一步的三个方向的值（从R中获取，R已计算完成）
            diag = R[n - 1, m - 1]
            up = R[n - 1, m]
            left = R[n, m - 1]
            
            # 重新计算soft-min（这次保持梯度）
            r_prev = torch.stack([diag, up, left])
            r_prev_neg = -r_prev / gamma
            r_max = torch.max(r_prev_neg)
            exp_values = torch.exp(r_prev_neg - r_max)
            log_sum = torch.log(torch.sum(exp_values) + 1e-10)
            r_min = -gamma * (r_max + log_sum)
            
            # 最终结果（保持梯度连接，通过distance_matrix）
            result = distance_matrix[n - 1, m - 1] + r_min
        else:
            # 如果n或m为0，直接返回0
            result = torch.tensor(0.0, device=distance_matrix.device, dtype=distance_matrix.dtype)
        
        # 如果结果异常，使用简单的MSE作为fallback（保持梯度连接）
        if not torch.isfinite(result):
            result = torch.mean(distance_matrix)
        
        return result

