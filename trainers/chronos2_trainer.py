"""
Chronos-2 知识蒸馏训练器（特征蒸馏 + 预测蒸馏 + 可选DTW损失）

设计目标：
- 把原脚本里“模型搬设备、hook特征、loss组合、checkpoint保存/恢复、训练/评估循环”模块化
- 尽量复用项目中已存在的通用模块：
  - `TimeDistill.models.FeatureRegressor`
  - `TimeDistill.losses.DTWLoss`
  - `TimeDistill.utils.device_utils.move_model_to_device`
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from chronos import Chronos2Model, Chronos2Pipeline
from torch.utils.data import DataLoader

from losses import DTWLoss
from models import FeatureRegressor
from utils.device_utils import move_model_to_device


@dataclass
class Chronos2DistillLossWeights:
    """loss 权重配置"""

    alpha_pred_distill: float = 0.5  # 预测蒸馏(soft) vs 真实标签(hard) 的插值
    beta_feature: float = 0.3  # 特征蒸馏权重
    dtw_weight: float = 0.2  # DTW 损失权重（0 表示禁用）


class Chronos2DistillationTrainer:
    """Chronos-2 知识蒸馏训练器"""

    def __init__(
        self,
        teacher_pipeline: Chronos2Pipeline,
        student_model: Chronos2Model,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        horizon: int = 24,
        learning_rate: float = 5e-5,
        num_epochs: int = 10,
        temperature: float = 2.0,  # 预留：回归任务目前未显式使用 softmax 温度
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./chronos-2-distilled",
        max_grad_norm: float = 1.0,
        loss_weights: Chronos2DistillLossWeights = Chronos2DistillLossWeights(),
        dtw_gamma: float = 1.0,
        use_fast_dtw: bool = True,
        verbose: bool = True,
    ):
        self.teacher_pipeline = teacher_pipeline
        self.student_model = student_model
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.horizon = horizon
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.temperature = temperature
        self.device = device
        self.output_dir = output_dir
        self.max_grad_norm = max_grad_norm
        self.loss_weights = loss_weights
        self.verbose = verbose

        # DTW loss
        self.dtw_gamma = dtw_gamma
        self.dtw_loss_fn = DTWLoss(gamma=dtw_gamma, use_fast_approx=use_fast_dtw).to(device)

        # 设备与冻结 teacher
        self._log(f"将教师模型移动到设备: {device}")
        self.teacher_pipeline.model = move_model_to_device(self.teacher_pipeline.model, device)
        self._log(f"将学生模型移动到设备: {device}")
        self.student_model = move_model_to_device(self.student_model, device)

        self.teacher_pipeline.model.eval()
        for p in self.teacher_pipeline.model.parameters():
            p.requires_grad = False
        self._log("教师模型已设置为评估模式并冻结所有参数")

        # 特征回归器（维度对齐）
        teacher_dim = self.teacher_pipeline.model.config.d_model
        student_dim = self.student_model.config.d_model
        self.feature_regressor_first = FeatureRegressor(teacher_dim, student_dim).to(device)
        self.feature_regressor_last = FeatureRegressor(teacher_dim, student_dim).to(device)
        self._log(f"创建特征回归器: 教师维度={teacher_dim}, 学生维度={student_dim}")

        # hook 缓存
        self.teacher_features: Dict[str, torch.Tensor] = {}
        self.student_features: Dict[str, torch.Tensor] = {}
        self.teacher_handles = []
        self.student_handles = []
        self._register_hooks()

        # optimizer（包含学生模型 + regressors）
        optimizer_params = (
            list(self.student_model.parameters())
            + list(self.feature_regressor_first.parameters())
            + list(self.feature_regressor_last.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=learning_rate,
            weight_decay=1e-4,
            eps=1e-8,
        )

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # --------------------
    # hooks
    # --------------------
    def _register_hooks(self):
        """注册 hook：提取 teacher/student 第一层、最后一层 encoder block 的 hidden_states"""
        self._remove_hooks()

        teacher_model = self.teacher_pipeline.model
        student_model = self.student_model

        teacher_num_layers = len(teacher_model.encoder.block)
        student_num_layers = len(student_model.encoder.block)
        if teacher_num_layers <= 0 or student_num_layers <= 0:
            raise ValueError("encoder.block 层数为0，无法注册特征hook")

        teacher_first_layer = teacher_model.encoder.block[0]
        teacher_last_layer = teacher_model.encoder.block[teacher_num_layers - 1]
        student_first_layer = student_model.encoder.block[0]
        student_last_layer = student_model.encoder.block[student_num_layers - 1]

        def _extract_hidden_states(output: Any) -> torch.Tensor:
            if hasattr(output, "hidden_states"):
                return output.hidden_states
            if isinstance(output, tuple):
                return output[0]
            return output

        def teacher_first_hook(_module, _input, output):
            self.teacher_features["first"] = _extract_hidden_states(output)

        def teacher_last_hook(_module, _input, output):
            self.teacher_features["last"] = _extract_hidden_states(output)

        def student_first_hook(_module, _input, output):
            self.student_features["first"] = _extract_hidden_states(output)

        def student_last_hook(_module, _input, output):
            self.student_features["last"] = _extract_hidden_states(output)

        self.teacher_handles = [
            teacher_first_layer.register_forward_hook(teacher_first_hook),
            teacher_last_layer.register_forward_hook(teacher_last_hook),
        ]
        self.student_handles = [
            student_first_layer.register_forward_hook(student_first_hook),
            student_last_layer.register_forward_hook(student_last_hook),
        ]
        self._log("已注册特征提取hook（第一层和最后一层）")

    def _remove_hooks(self):
        for h in getattr(self, "teacher_handles", []):
            try:
                h.remove()
            except Exception:
                pass
        for h in getattr(self, "student_handles", []):
            try:
                h.remove()
            except Exception:
                pass
        self.teacher_handles = []
        self.student_handles = []

    # --------------------
    # loss
    # --------------------
    def _compute_teacher_student_logits(
        self, context: torch.Tensor, group_ids: torch.Tensor, num_output_patches: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 (teacher_logits, student_logits)，均为 [B, horizon]，必要时截断/填充到 horizon。
        """
        batch_size = context.shape[0]

        # teacher forward
        with torch.no_grad():
            teacher_out = self.teacher_pipeline.model(
                context=context,
                group_ids=group_ids,
                num_output_patches=num_output_patches,
            )
            teacher_pred = teacher_out.quantile_preds  # [B, Q, L]
            median_idx = teacher_pred.shape[1] // 2
            teacher_logits = teacher_pred[:, median_idx, :]
            teacher_logits = self._pad_or_trim_to_horizon(teacher_logits, batch_size=batch_size)

        # student forward
        student_out = self.student_model(
            context=context,
            group_ids=group_ids,
            num_output_patches=num_output_patches,
        )
        student_pred = student_out.quantile_preds
        median_idx = student_pred.shape[1] // 2
        student_logits = student_pred[:, median_idx, :]
        student_logits = self._pad_or_trim_to_horizon(student_logits, batch_size=batch_size)

        return teacher_logits, student_logits

    def _pad_or_trim_to_horizon(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        if x.shape[1] > self.horizon:
            return x[:, : self.horizon]
        if x.shape[1] < self.horizon:
            pad = torch.zeros(batch_size, self.horizon - x.shape[1], device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=1)
        return x

    def _compute_losses(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        future_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # 对齐长度
        min_len = min(student_logits.shape[1], teacher_logits.shape[1], future_target.shape[1])
        student_logits = student_logits[:, :min_len]
        teacher_logits = teacher_logits[:, :min_len]
        future_target = future_target[:, :min_len]

        # 基础 loss
        mse_hard = F.mse_loss(student_logits, future_target)
        mse_soft = F.mse_loss(student_logits, teacher_logits)

        # 特征蒸馏（first/last）
        feat_loss = torch.tensor(0.0, device=self.device)
        feat_first = torch.tensor(0.0, device=self.device)
        feat_last = torch.tensor(0.0, device=self.device)

        if "first" in self.teacher_features and "first" in self.student_features:
            t = self.feature_regressor_first(self.teacher_features["first"])
            s = self.student_features["first"]
            feat_first = F.mse_loss(s, t)
            feat_loss = feat_loss + feat_first

        if "last" in self.teacher_features and "last" in self.student_features:
            t = self.feature_regressor_last(self.teacher_features["last"])
            s = self.student_features["last"]
            feat_last = F.mse_loss(s, t)
            feat_loss = feat_loss + feat_last

        # DTW
        dtw = torch.tensor(0.0, device=self.device)
        if self.loss_weights.dtw_weight > 0 and min_len > 0:
            dtw = self.dtw_loss_fn(student_logits, future_target)
            dtw = torch.clamp(dtw, min=0.0)

        # total
        w = self.loss_weights
        total = (
            w.alpha_pred_distill * mse_soft
            + (1.0 - w.alpha_pred_distill) * mse_hard
            + w.beta_feature * feat_loss
            + w.dtw_weight * dtw
        )

        return {
            "total": total,
            "mse_hard": mse_hard,
            "mse_soft": mse_soft,
            "feat": feat_loss,
            "feat_first": feat_first,
            "feat_last": feat_last,
            "dtw": dtw,
        }

    # --------------------
    # train / eval
    # --------------------
    def train_epoch(self, epoch: int) -> float:
        self.student_model.train()
        total = 0.0
        n = 0

        for batch_idx, batch in enumerate(self.train_loader):
            context = batch["target"].to(self.device)  # [B, context_len]
            future_target = batch["future_target"].to(self.device)  # [B, horizon]
            batch_size = context.shape[0]

            output_patch_size = self.student_model.chronos_config.output_patch_size
            num_output_patches = (self.horizon + output_patch_size - 1) // output_patch_size
            group_ids = torch.arange(batch_size, device=self.device)

            # 清空特征缓存（由 hook 填充）
            self.teacher_features = {}
            self.student_features = {}

            self.optimizer.zero_grad(set_to_none=True)

            try:
                teacher_logits, student_logits = self._compute_teacher_student_logits(
                    context=context, group_ids=group_ids, num_output_patches=num_output_patches
                )
            except Exception as e:
                self._log(f"警告: 前向传播失败，跳过 batch={batch_idx}: {e}")
                continue

            if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
                self._log(f"警告: logits 包含 NaN/Inf，跳过 batch={batch_idx}")
                continue

            losses = self._compute_losses(
                student_logits=student_logits, teacher_logits=teacher_logits, future_target=future_target
            )
            loss = losses["total"]

            if (not torch.isfinite(loss)) or loss.item() < 0 or loss.item() > 1e6:
                self._log(
                    f"警告: loss 异常，跳过 batch={batch_idx}, "
                    f"loss={loss.item():.6f}, mse_hard={losses['mse_hard'].item():.6f}, "
                    f"mse_soft={losses['mse_soft'].item():.6f}, feat={losses['feat'].item():.6f}, dtw={losses['dtw'].item():.6f}"
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total += loss.item()
            n += 1

            if self.verbose and (batch_idx + 1) % 10 == 0:
                self._log(
                    f"Epoch {epoch}, Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"loss={loss.item():.4f} hard={losses['mse_hard'].item():.4f} soft={losses['mse_soft'].item():.4f} "
                    f"feat={losses['feat'].item():.4f} dtw={losses['dtw'].item():.4f}"
                )

        return total / n if n > 0 else 0.0

    @torch.no_grad()
    def evaluate(self) -> Optional[Dict[str, float]]:
        if self.eval_loader is None:
            return None

        self.student_model.eval()
        total_hard = 0.0
        total_dtw = 0.0
        n = 0

        for batch in self.eval_loader:
            context = batch["target"].to(self.device)
            future_target = batch["future_target"].to(self.device)
            batch_size = context.shape[0]

            output_patch_size = self.student_model.chronos_config.output_patch_size
            num_output_patches = (self.horizon + output_patch_size - 1) // output_patch_size
            group_ids = torch.arange(batch_size, device=self.device)

            # 清空缓存
            self.teacher_features = {}
            self.student_features = {}

            try:
                teacher_logits, student_logits = self._compute_teacher_student_logits(
                    context=context, group_ids=group_ids, num_output_patches=num_output_patches
                )
            except Exception:
                continue

            min_len = min(student_logits.shape[1], future_target.shape[1])
            mse_hard = F.mse_loss(student_logits[:, :min_len], future_target[:, :min_len])
            total_hard += mse_hard.item()

            if self.loss_weights.dtw_weight > 0 and min_len > 0:
                try:
                    dtw = self.dtw_loss_fn(student_logits[:, :min_len], future_target[:, :min_len])
                    total_dtw += float(dtw.item())
                except Exception:
                    pass

            n += 1

        if n == 0:
            return {"mse_loss": 0.0, "dtw_loss": 0.0}

        return {"mse_loss": total_hard / n, "dtw_loss": total_dtw / n}

    def train(
        self,
        resume_from_epoch: Optional[int] = None,
        resume_checkpoint_path: Optional[str] = None,
    ):
        os.makedirs(self.output_dir, exist_ok=True)
        self._log("=" * 50)
        self._log("开始 Chronos-2 蒸馏训练")
        self._log("=" * 50)

        start_epoch = 0
        best_eval = float("inf")

        # resume
        if resume_checkpoint_path:
            start_epoch, best_eval = self.load_checkpoint(resume_checkpoint_path)
            self._log(f"从 epoch={start_epoch+1} 继续训练，best_eval={best_eval:.6f}")
        elif resume_from_epoch is not None:
            ckpt = os.path.join(self.output_dir, f"checkpoint_epoch_{resume_from_epoch}")
            if os.path.exists(ckpt):
                start_epoch, best_eval = self.load_checkpoint(ckpt)
                self._log(f"从 epoch={start_epoch+1} 继续训练，best_eval={best_eval:.6f}")
            else:
                raise FileNotFoundError(f"未找到 checkpoint: {ckpt}")

        for epoch in range(start_epoch, self.num_epochs):
            self._log(f"\nEpoch {epoch+1}/{self.num_epochs}")
            self._log("-" * 50)

            train_loss = self.train_epoch(epoch)
            self._log(f"训练损失: {train_loss:.6f}")

            if self.eval_loader is not None:
                metrics = self.evaluate()
                if metrics:
                    self._log(f"验证 MSE: {metrics['mse_loss']:.6f}, DTW: {metrics['dtw_loss']:.6f}")
                    if metrics["mse_loss"] < best_eval:
                        best_eval = metrics["mse_loss"]
                        self.save_model(f"best_model_epoch_{epoch+1}")
                        self._log(f"保存最佳模型: best_eval={best_eval:.6f}")

            self.save_checkpoint(epoch=epoch, best_eval_loss=best_eval, checkpoint_name="checkpoint")

        self.save_model("final_model")
        self._log("\n训练完成！")

    # --------------------
    # checkpointing
    # --------------------
    def save_model(self, model_name: str):
        save_path = os.path.join(self.output_dir, model_name)
        os.makedirs(save_path, exist_ok=True)
        self.student_model.save_pretrained(save_path)
        self._log(f"模型已保存到: {save_path}")

    def save_checkpoint(self, epoch: int, best_eval_loss: float, checkpoint_name: str = "checkpoint") -> str:
        checkpoint_path = os.path.join(self.output_dir, f"{checkpoint_name}_epoch_{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # model
        model_path = os.path.join(checkpoint_path, "model")
        os.makedirs(model_path, exist_ok=True)
        self.student_model.save_pretrained(model_path)

        # regressors
        torch.save(
            {
                "feature_regressor_first": self.feature_regressor_first.state_dict(),
                "feature_regressor_last": self.feature_regressor_last.state_dict(),
            },
            os.path.join(checkpoint_path, "regressors.pt"),
        )

        # training state
        torch.save(
            {
                "epoch": epoch,
                "best_eval_loss": best_eval_loss,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "temperature": self.temperature,
                "loss_weights": self.loss_weights.__dict__,
                "dtw_gamma": self.dtw_gamma,
            },
            os.path.join(checkpoint_path, "training_state.pt"),
        )

        self._log(f"Checkpoint已保存到: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, float]:
        model_path = os.path.join(checkpoint_path, "model")
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        regressor_path = os.path.join(checkpoint_path, "regressors.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型目录不存在: {model_path}")
        if not os.path.exists(training_state_path):
            raise FileNotFoundError(f"训练状态文件不存在: {training_state_path}")

        self._log(f"从checkpoint加载: {checkpoint_path}")

        # model
        self._log("加载学生模型...")
        self.student_model = Chronos2Model.from_pretrained(model_path)
        self.student_model = move_model_to_device(self.student_model, self.device)

        # regressors
        if os.path.exists(regressor_path):
            self._log("加载特征回归器...")
            state = torch.load(regressor_path, map_location=self.device)
            self.feature_regressor_first.load_state_dict(state["feature_regressor_first"])
            self.feature_regressor_last.load_state_dict(state["feature_regressor_last"])

        # state
        self._log("加载训练状态...")
        checkpoint = torch.load(training_state_path, map_location=self.device)
        epoch = int(checkpoint.get("epoch", 0))
        best_eval = float(checkpoint.get("best_eval_loss", float("inf")))

        # rebuild optimizer with new params
        optimizer_params = (
            list(self.student_model.parameters())
            + list(self.feature_regressor_first.parameters())
            + list(self.feature_regressor_last.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=float(checkpoint.get("learning_rate", self.learning_rate)),
            weight_decay=1e-4,
            eps=1e-8,
        )
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self._log("Optimizer状态已加载")
            except Exception as e:
                self._log(f"警告: 无法加载optimizer状态，将使用新的optimizer状态: {e}")

        # restore loss weights if present
        lw = checkpoint.get("loss_weights", None)
        if isinstance(lw, dict):
            self.loss_weights = Chronos2DistillLossWeights(**lw)

        self._log(f"成功加载checkpoint: epoch={epoch+1}, best_eval={best_eval:.6f}")
        return epoch, best_eval


