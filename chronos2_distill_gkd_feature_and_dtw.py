"""
（兼容入口）Chronos-2 蒸馏脚本 - 特征蒸馏 + DTW

原先该文件包含完整实现（>1000 行）。为便于复用与维护，已将各模块解耦并重构到：
- 数据集：`TimeDistill.datasets.Chronos2DistillationDataset`
- 训练器：`TimeDistill.trainers.Chronos2DistillationTrainer`
- DTW：`TimeDistill.losses.DTWLoss`
- 特征回归器：`TimeDistill.models.FeatureRegressor`
- 入口脚本：`TimeDistill/scripts/chronos2_distill_feature_dtw.py`

如需运行，请直接执行：
  python -m TimeDistill.scripts.chronos2_distill_feature_dtw
"""

from __future__ import annotations

from scripts.chronos2_distill_feature_dtw import main


if __name__ == "__main__":
    main()


