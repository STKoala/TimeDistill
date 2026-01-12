"""
蒸馏脚本 - 使用 TRL GKD 的适配方案

由于 Chronos-2 使用自定义架构（非标准 Transformer），直接使用 TRL GKD 可能不兼容。
本脚本提供了一个可行的适配方案：

方案 1：使用 Chronos-2 的 fit 方法 + 手动蒸馏损失（推荐）

本脚本实现方案 1，这是最稳定和可行的方案。

后续逐步添加选项，可选择蒸馏 timesfm、chronos-2、chronos-bolt、timemoe、
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import TrainingArguments, Trainer
from chronos import Chronos2Pipeline, Chronos2Model
from chronos.chronos2 import Chronos2ForecastingConfig, Chronos2CoreConfig
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
from sklearn.preprocessing import StandardScaler
import warnings