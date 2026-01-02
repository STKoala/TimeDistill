"""
PatchTST 学生模型蒸馏 Chronos-2 教师模型的示例脚本。

主要步骤：
- 使用 Chronos2Pipeline 生成教师预测，缓存为监督信号。
- 使用简化版 PatchTST 学生（PyTorch 实现）回归教师预测。
- 默认读取 `datasets/ETTh1.csv`，目标列优先选择 OT。
"""
import os
import argparse
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import set_seed
from tqdm import tqdm

# ----------------------------
# 数据集与 Chronos 教师输出缓存
# ----------------------------
class TeacherCacheDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def prepare_context_df(df: pd.DataFrame, start_idx: int, context_len: int, target_col: str):
    """Chronos2Pipeline 需要 id、timestamp、target 结构。"""
    context = df.iloc[start_idx : start_idx + context_len]
    return pd.DataFrame(
        {
            "id": ["series_0"] * len(context),
            "timestamp": pd.to_datetime(context["date"]),
            "target": context[target_col].values,
        }
    )


def predict_with_chronos(pipeline, context_df: pd.DataFrame, horizon: int) -> np.ndarray:
    """统一从 Chronos2Pipeline 取出中位数预测。"""
    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=horizon,
        quantile_levels=[0.5],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )
    for key in ["0.5", "target", "mean"]:
        if key in pred_df.columns:
            return pred_df[key].values.astype(np.float32)
    numeric_cols = [
        c
        for c in pred_df.columns
        if c not in {"id", "timestamp", "item_id", "time_idx"}
        and pd.api.types.is_numeric_dtype(pred_df[c])
    ]
    if not numeric_cols:
        raise ValueError("Chronos 预测结果中未找到数值列")
    return pred_df[numeric_cols[0]].values.astype(np.float32)


def _collect_one_sample(df: pd.DataFrame, pipeline, cfg, start_idx: int, debug_state: dict):
    """单个起点的教师预测收集，便于并行调度。"""
    context_df = prepare_context_df(df, start_idx, cfg.context_len, cfg.target_col)
    teacher_pred = predict_with_chronos(pipeline, context_df, cfg.horizon)
    if len(teacher_pred) < cfg.horizon:
        return None
    context_values = (
        df[cfg.target_col]
        .iloc[start_idx : start_idx + cfg.context_len]
        .values.astype(np.float32)
    )
    if cfg.debug_samples and debug_state["shown"] < cfg.debug_samples:
        gt = df[cfg.target_col].iloc[
            start_idx + cfg.context_len : start_idx + cfg.context_len + cfg.horizon
        ].values.astype(np.float32)
        print(
            f"[DEBUG] start={start_idx} | context_tail={context_values[-3:].tolist()} | "
            f"teacher_pred[:5]={teacher_pred[:5].tolist()} | gt[:5]={gt[:5].tolist()}"
        )
        debug_state["shown"] += 1
    return context_values, teacher_pred[: cfg.horizon]


def iter_teacher_cache(df: pd.DataFrame, pipeline, cfg, desc: str):
    """生成器：按滑窗产出样本，支持线程并行。"""
    total = len(df)
    max_start = total - cfg.context_len - cfg.horizon
    step = max(1, cfg.cache_stride)
    indices = list(range(0, max_start + 1, step))
    debug_state = {"shown": 0}

    if cfg.cache_workers > 0:
        with ThreadPoolExecutor(max_workers=cfg.cache_workers) as ex:
            futures = {ex.submit(_collect_one_sample, df, pipeline, cfg, idx, debug_state): idx for idx in indices}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                res = fut.result()
                if res is not None:
                    yield res
    else:
        for start_idx in tqdm(indices, desc=desc):
            res = _collect_one_sample(df, pipeline, cfg, start_idx, debug_state)
            if res is not None:
                yield res


def build_teacher_cache(df: pd.DataFrame, pipeline, cfg) -> List[Tuple[np.ndarray, np.ndarray]]:
    """滑窗生成 (context, teacher_pred) 样本。"""
    samples: List[Tuple[np.ndarray, np.ndarray]] = []
    for res in iter_teacher_cache(df, pipeline, cfg, desc="收集教师预测"):
        samples.append(res)
    return samples


# ----------------------------
# 简化版 PatchTST 学生模型
# ----------------------------
def positional_encoding(pe: str, learn_pe: bool, num_patch: int, d_model: int):
    """简化版位置编码：零初始化可学习/固定。"""
    if pe == "zeros":
        w_pos = torch.zeros(1, num_patch, d_model)
    else:
        w_pos = torch.randn(1, num_patch, d_model) * 0.02
    return nn.Parameter(w_pos, requires_grad=learn_pe)


class SimpleTSTEncoder(nn.Module):
    """用 nn.TransformerEncoder 近似 TSTEncoder。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        attn_dropout: float,
        dropout: float,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        # x: [B, num_patch, d_model]
        return self.encoder(x)


class PatchTSTStudent(nn.Module):
    """
    简化版 PatchTSTEncoder：单变量 (c_in=1) 情况，按参考结构拆 patch -> 位置编码 -> Transformer。
    """

    def __init__(
        self,
        input_len: int,
        horizon: int,
        patch_len: int = 32,
        stride: int = 8,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        d_ff: int = 256,
        pe: str = "zeros",
        learn_pe: bool = True,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.horizon = horizon
        self.c_in = 1  # 单变量

        num_patch = (input_len - patch_len) // stride + 1

        # shared embedding
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder = SimpleTSTEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=num_layers,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x: torch.Tensor):
        # 期望输入 [B, T] 或 [B, T, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, T, 1]

        # 构造 patch -> [B, num_patch, c_in, patch_len]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # [B, num_patch, c_in, patch_len]

        # shared embedding
        x1 = self.W_P(patches)  # [B, num_patch, c_in, d_model]
        # reshape to [B * c_in, num_patch, d_model]
        b, num_patch, c_in, _ = x1.shape
        u = x1.view(b * c_in, num_patch, -1)
        u = self.dropout(u + self.W_pos)

        z = self.encoder(u)  # [B * c_in, num_patch, d_model]
        z = z.view(b, c_in, num_patch, -1)  # [B, c_in, num_patch, d_model]
        z = z.permute(0, 1, 3, 2)  # [B, c_in, d_model, num_patch]

        # 简化：对变量和时间维做平均池化
        pooled = z.mean(dim=(1, 3))  # [B, d_model]
        out = self.head(pooled)  # [B, horizon]
        return out


# ----------------------------
# 训练逻辑
# ----------------------------
@dataclass
class DistillConfig:
    data_path: str = "datasets/ETTh1.csv"
    data_paths: Optional[str] = None  # 逗号分隔多个数据集
    target_col: str = "OT"
    context_len: int = 96
    horizon: int = 24
    cache_stride: int = 24  # 教师滑窗步长
    debug_samples: int = 0  # 打印前 n 个样本的教师预测与真值
    cache_stage_size: int = 0  # >0 时，分批收集 cache 并立即训练
    stage_epochs: int = 1      # 分批训练时，每批迭代轮数
    cache_workers: int = 0     # >0 时，生成教师 cache 线程并行
    patch_len: int = 32
    stride: int = 8
    d_model: int = 256
    n_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = False
    lr: float = 1e-3
    epochs: int = 5
    distill_alpha: float = 0.5  # KL(teacher logits) vs student logits 权重
    distill_temp: float = 2.0   # 温度系数
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "patchtst_student.pt"


def train_student(
    dataset: Dataset,
    cfg: DistillConfig,
    model: Optional[PatchTSTStudent] = None,
    optim: Optional[torch.optim.Optimizer] = None,
    base_epoch: int = 0,
    epochs: Optional[int] = None,
    save_on_finish: bool = True,
):
    """可复用模型/优化器的训练函数，支持分批训练。"""
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
    )
    if model is None:
        model = PatchTSTStudent(
            input_len=cfg.context_len,
            horizon=cfg.horizon,
            patch_len=cfg.patch_len,
            stride=cfg.stride,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(cfg.device)
    if optim is None:
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    cur_epochs = epochs or cfg.epochs

    for local_epoch in range(cur_epochs):
        model.train()
        total = 0.0
        count = 0
        for x, teacher_y in loader:
            x = x.to(cfg.device)
            teacher_y = teacher_y.to(cfg.device)
            pred = model(x)
            # 1) value 蒸馏（回归教师输出）
            mse = mse_loss(pred, teacher_y)
            # 2) logits 蒸馏（对输出作 softmax 后做 KL）
            t = cfg.distill_temp
            student_logit = pred / t
            teacher_logit = teacher_y / t
            kl = kl_loss(
                F.log_softmax(student_logit, dim=-1),
                F.softmax(teacher_logit, dim=-1),
            ) * (t * t)
            loss = cfg.distill_alpha * kl + (1 - cfg.distill_alpha) * mse
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
        global_epoch = base_epoch + local_epoch + 1
        print(f"Epoch {global_epoch}/{base_epoch + cur_epochs} | train_mse={total / count:.6f}")

    if save_on_finish:
        torch.save(model.state_dict(), cfg.save_path)
        print(f"保存学生权重到 {cfg.save_path}")
    return model, optim, base_epoch + cur_epochs


def main():
    from chronos import Chronos2Pipeline

    parser = argparse.ArgumentParser(description="PatchTST 蒸馏 Chronos-2")
    parser.add_argument("--data_path", type=str, default="datasets/ETTh1.csv")
    parser.add_argument("--data_paths", type=str, default=None, help="逗号分隔，覆盖 data_path，支持多数据集")
    parser.add_argument("--target_col", type=str, default="OT")
    parser.add_argument("--context_len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--cache_stride", type=int, default=24, help="教师滑窗步长，默认等于 horizon")
    parser.add_argument("--patch_len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--distill_alpha", type=float, default=0.5, help="kl 与 mse 的权重，1 表示只用 kl")
    parser.add_argument("--distill_temp", type=float, default=2.0, help="logits 蒸馏温度")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto", help="cuda|cpu|auto")
    parser.add_argument("--save_path", type=str, default="patchtst_student.pt")
    parser.add_argument("--debug_samples", type=int, default=0, help="打印前 n 个样本的教师预测与真值")
    parser.add_argument("--cache_stage_size", type=int, default=0, help=">0 时启用分批收集 cache 的训练；值为每批样本数")
    parser.add_argument("--stage_epochs", type=int, default=1, help="分批训练时每批迭代轮数")
    parser.add_argument("--cache_workers", type=int, default=0, help=">0 时教师预测收集使用线程并行")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 线程数")
    parser.add_argument("--pin_memory", action="store_true", help="DataLoader 启用 pin_memory")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader 预取因子（num_workers>0 时生效）")
    parser.add_argument("--persistent_workers", action="store_true", help="DataLoader 持续工作线程（num_workers>0 时推荐）")
    args = parser.parse_args()

    auto_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    cfg = DistillConfig(
        data_path=args.data_path,
        data_paths=args.data_paths,
        target_col=args.target_col,
        context_len=args.context_len,
        horizon=args.horizon,
        cache_stride=args.cache_stride,
        debug_samples=args.debug_samples,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        lr=args.lr,
        epochs=args.epochs,
        distill_alpha=args.distill_alpha,
        distill_temp=args.distill_temp,
        seed=args.seed,
        device=auto_device,
        save_path=args.save_path,
        cache_stage_size=args.cache_stage_size,
        stage_epochs=args.stage_epochs,
        cache_workers=args.cache_workers,
    )
    set_seed(cfg.seed)
    print(f"使用设备: {cfg.device}")

    # 解析数据集列表
    data_paths = []
    if cfg.data_paths:
        data_paths = [p.strip() for p in cfg.data_paths.split(",") if p.strip()]
    if not data_paths:
        data_paths = [cfg.data_path]

    print(f"加载 Chronos-2 教师模型...")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=cfg.device)

    all_samples: List[Tuple[np.ndarray, np.ndarray]] = []
    model: Optional[PatchTSTStudent] = None
    optim: Optional[torch.optim.Optimizer] = None
    trained_epochs = 0

    for path in data_paths:
        if not os.path.exists(path):
            print(f"警告: 未找到数据文件 {path}，跳过")
            continue
        df = pd.read_csv(path)
        if cfg.target_col not in df.columns:
            cfg.target_col = df.columns[-1]
            print(f"{path}: 未找到 OT 列，使用最后一列作为目标: {cfg.target_col}")
        print(f"{path}: 开始生成教师预测缓存 (context={cfg.context_len}, horizon={cfg.horizon}, step={cfg.cache_stride})")

        # 若 cache_stage_size > 0，边收集边训练，避免全部缓存占用。
        if cfg.cache_stage_size > 0:
            stage_samples: List[Tuple[np.ndarray, np.ndarray]] = []
            for sample in iter_teacher_cache(df, pipeline, cfg, desc="收集教师预测-分批"):
                stage_samples.append(sample)
                if len(stage_samples) >= cfg.cache_stage_size:
                    dataset = TeacherCacheDataset(stage_samples)
                    model, optim, trained_epochs = train_student(
                        dataset,
                        cfg,
                        model=model,
                        optim=optim,
                        base_epoch=trained_epochs,
                        epochs=cfg.stage_epochs,
                        save_on_finish=False,
                    )
                    stage_samples = []

            # 处理尾批
            if stage_samples:
                dataset = TeacherCacheDataset(stage_samples)
                model, optim, trained_epochs = train_student(
                    dataset,
                    cfg,
                    model=model,
                    optim=optim,
                    base_epoch=trained_epochs,
                    epochs=cfg.stage_epochs,
                    save_on_finish=False,
                )
        else:
            print("cache_stage_size: 0")
            samples = build_teacher_cache(df, pipeline, cfg)
            all_samples.extend(samples)

    if cfg.cache_stage_size > 0:
        if model is None:
            raise RuntimeError("未生成任何教师样本，请检查数据路径或参数。")
        torch.save(model.state_dict(), cfg.save_path)
        print(f"保存学生权重到 {cfg.save_path}")
    else:
        if not all_samples:
            raise RuntimeError("未生成任何教师样本，请检查数据路径或参数。")
        dataset = TeacherCacheDataset(all_samples)
        print(f"可用训练样本总数: {len(dataset)}")
        train_student(dataset, cfg)


if __name__ == "__main__":
    main()

