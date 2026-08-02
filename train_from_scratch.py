"""Train the WaveU-Net + Transformer denoising model from scratch.

本脚本从 ``GW_A100_Checkpoints2.ipynb`` 中的 "Training from scratch" 部分抽离而来，
将所需的依赖（自定义 Dataset、掩码与标准化工具、模型定义、掩码损失、检查点管理器
以及带检查点的训练循环）整合到同一个文件中，方便在 A100 等 HPC 节点直接以脚本方式
后台运行。

使用示例（默认参数等价于 notebook 中的 "First trained with SNR 200" 配置）：

    python train_from_scratch.py \
        --data-folder ./waveforms2 \
        --target-snr 200.00 \
        --num-epochs 200 \
        --batch-size 32 \
        --num-workers 24 \
        --learning-rate 1e-5 \
        --weight-decay 1e-6 \
        --save-every-n-epochs 10 \
        --checkpoint-dir ./model_checkpoints

如需先按 SNR=200 训练再继续用 SNR=50, 100 微调，请使用对应的
``Continue training from best_model or resume from checkpoint`` 单元的脚本
（或在本脚本基础上指定 ``--resume-from-checkpoint``）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import OrderedDict
from datetime import datetime

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GWDataset(Dataset):
    """新版数据结构：每个 .h5 文件包含大量 group，每个 group 是一条 waveform。

    Group 命名: ``waveform{i}_SNR{xx.xx}``，例如 ``waveform50000_SNR50.00``。
    Group 内部:
        - ``white_Data``    : (L,) 数据
        - ``white_signal``  : (L,) 信号
        - ``attrs``         : ``mc_true``/``phis_true``/``thetas_true``/``dl_true``/
                              ``eta_true``/``iota_true``/``phic_true``/``psi_true``/
                              ``tc_true``/``snr`` 等

    设计要点（针对 52 × ~26GB 的数据规模）:
        1. ``__init__`` 只构建轻量级索引 ``(file_idx, group_name, snr_str)``，
           **不加载任何波形数据**，并且每个文件只在首个匹配 group 上做结构校验。
        2. ``__getitem__`` 按需打开 h5 文件，使用每实例的 **LRU 缓存** 控制
           同时持有的句柄数（``max_open_files``），并通过 ``rdcc_nbytes`` 限制
           每个文件的 chunk cache，避免随机访问导致缓存膨胀。
        3. ``h5py.File`` 句柄不可跨进程 pickle，所以 ``__getstate__`` 中剔除句柄缓存，
           DataLoader 多 worker 时由各 worker 各自打开自己的副本。
    """

    GROUP_PATTERN = re.compile(r"^waveform\d+_SNR(\d+\.\d+)$")
    FILE_PATTERN = re.compile(r"^waveforms_proc(\d+)\.h5$")
    REQUIRED_ATTRS = ("mc_true", "phis_true", "thetas_true")

    def __init__(
        self,
        folder_path,
        snr_list=None,
        transform=None,
        max_open_files=8,
        rdcc_nbytes=1024 * 1024,
        verbose=True,
        file_pattern=None,
    ):
        self.folder_path = folder_path
        self.transform = transform
        self.snr_list = list(snr_list) if snr_list else ["50.00", "100.00", "200.00"]
        self.snr_set = set(self.snr_list)
        self.max_open_files = int(max_open_files)
        self.rdcc_nbytes = int(rdcc_nbytes)
        self.verbose = verbose
        self.file_pattern = (
            re.compile(file_pattern) if isinstance(file_pattern, str)
            else (file_pattern or self.FILE_PATTERN)
        )

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"数据目录不存在: {folder_path}")

        self.file_paths = []
        self.index = []

        h5_candidates = []
        for fname in os.listdir(folder_path):
            full_path = os.path.join(folder_path, fname)
            if not os.path.isfile(full_path):
                continue
            m = self.file_pattern.match(fname)
            if not m:
                continue
            try:
                key = int(m.group(1)) if m.groups() else fname
            except (IndexError, ValueError):
                key = fname
            h5_candidates.append((key, fname))

        h5_candidates.sort(key=lambda x: (isinstance(x[0], str), x[0]))
        h5_files = [fname for _, fname in h5_candidates]

        if self.verbose:
            print(f"匹配到 {len(h5_files)} 个候选文件 (pattern={self.file_pattern.pattern})")

        for fname in h5_files:
            full_path = os.path.join(folder_path, fname)

            if not os.access(full_path, os.R_OK):
                print(f"警告: 文件不可读，跳过: {full_path}")
                continue

            try:
                with h5py.File(full_path, "r") as f:
                    group_names = list(f.keys())
                    if not group_names:
                        print(f"警告: 文件无任何 group，跳过: {full_path}")
                        continue

                    file_idx = len(self.file_paths)
                    self.file_paths.append(full_path)

                    matched = 0
                    sample_validated = False
                    file_invalid = False

                    for gname in group_names:
                        m = self.GROUP_PATTERN.match(gname)
                        if not m:
                            continue
                        snr = m.group(1)
                        if snr not in self.snr_set:
                            continue

                        if not sample_validated:
                            g = f[gname]
                            if "white_Data" not in g or "white_signal" not in g:
                                print(
                                    f"警告: group {gname} 缺少 white_Data/white_signal，"
                                    f"跳过整个文件: {full_path}"
                                )
                                file_invalid = True
                                break
                            if not all(attr in g.attrs for attr in self.REQUIRED_ATTRS):
                                print(
                                    f"警告: group {gname} 缺少必要属性 {self.REQUIRED_ATTRS}，"
                                    f"跳过整个文件: {full_path}"
                                )
                                file_invalid = True
                                break
                            sample_validated = True

                        self.index.append((file_idx, gname, snr))
                        matched += 1

                    if file_invalid or matched == 0:
                        self.file_paths.pop()
                        if not file_invalid and self.verbose:
                            print(f"  - {fname}: 未匹配到任何指定 SNR 的 group，跳过")
                    elif self.verbose:
                        print(f"  ✓ {fname}: {matched} 个 waveform")

            except (OSError, IOError) as e:
                print(f"文件打开错误 {full_path}: {e}，跳过")
            except Exception as e:
                print(f"处理文件时出错 {full_path}: {e}，跳过")

        if not self.index:
            raise ValueError("没有匹配到任何指定 SNR 的 group，请检查路径或 snr_list 设置")

        print(
            f"成功加载 {len(self.file_paths)} 个文件，"
            f"共 {len(self.index)} 个 waveform 样本"
        )

        self._file_handles = None

    def _get_handle(self, file_idx):
        if self._file_handles is None:
            self._file_handles = OrderedDict()

        cache = self._file_handles
        if file_idx in cache:
            cache.move_to_end(file_idx)
            return cache[file_idx]

        path = self.file_paths[file_idx]
        f = h5py.File(
            path,
            mode="r",
            libver="latest",
            swmr=False,
            rdcc_nbytes=self.rdcc_nbytes,
        )
        cache[file_idx] = f

        while len(cache) > self.max_open_files:
            _, old_f = cache.popitem(last=False)
            try:
                old_f.close()
            except Exception:
                pass

        return f

    def close(self):
        if self._file_handles:
            for f in self._file_handles.values():
                try:
                    f.close()
                except Exception:
                    pass
        self._file_handles = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_handles"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file_handles = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, group_name, snr = self.index[idx]
        file_path = self.file_paths[file_idx]

        try:
            f = self._get_handle(file_idx)
            g = f[group_name]
            white_data = torch.from_numpy(g["white_Data"][:]).to(torch.float32)
            white_signal = torch.from_numpy(g["white_signal"][:]).to(torch.float32)
            attrs = {k: g.attrs[k] for k in g.attrs.keys()}
        except (OSError, IOError, KeyError) as e:
            print(f"读取错误 {file_path}::{group_name}: {e}")
            seq_len = 6184
            white_data = torch.zeros(seq_len, dtype=torch.float32)
            white_signal = torch.zeros(seq_len, dtype=torch.float32)
            attrs = {"mc_true": 0.0, "phis_true": 0.0, "thetas_true": 0.0}

        sample = {
            "white_data": white_data,
            "white_signal": white_signal,
            "attributes": attrs,
            "mc_true": float(attrs.get("mc_true", 0.0)),
            "phis_true": float(attrs.get("phis_true", 0.0)),
            "thetas_true": float(attrs.get("thetas_true", 0.0)),
            "snr": snr,
            "filename": os.path.basename(file_path),
            "group_name": group_name,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# ---------------------------------------------------------------------------
# Mask / Standardization helpers
# ---------------------------------------------------------------------------
def generate_mask(data: torch.Tensor, threshold_factor: float = 5.0) -> torch.Tensor:
    """构建掩码: 1 表示弱信号, 0 表示强信号, 2 表示第二段弱信号。"""
    B, L = data.shape
    mask = torch.ones_like(data, dtype=torch.long)
    stds = torch.std(data, dim=1, keepdim=True)
    thresholds = threshold_factor * stds

    abs_data = data.abs()
    for i in range(B):
        above_th = (abs_data[i] > thresholds[i])
        strong_indices = torch.nonzero(above_th).squeeze()

        if strong_indices.numel() > 0:
            if strong_indices.dim() == 0:
                start = end = strong_indices.item()
            else:
                start = strong_indices[0].item()
                end = strong_indices[-1].item()

            mask[i, start:end + 1] = 0
            mask[i, end + 1:] = 2

    return mask


def standardize_batch(
    data: torch.Tensor,
    signal: torch.Tensor,
    mask: torch.Tensor,
    amplification: float = 1.0,
):
    """RobustScaler 模拟：基于弱信号区域的中位数与 IQR 做标准化。"""
    signal_amplified = signal.clone()
    signal_amplified[mask == 1] *= amplification

    weak_indices = (mask == 1)
    weak_values = data[weak_indices].view(-1)
    median = weak_values.median()
    q1 = weak_values.kthvalue(int(len(weak_values) * 0.25))[0]
    q3 = weak_values.kthvalue(int(len(weak_values) * 0.75))[0]
    iqr = q3 - q1 + 1e-8

    data_std = (data - median) / iqr
    signal_std = (signal_amplified - median) / iqr

    stats = {"median": median.item(), "iqr": iqr.item()}
    return data_std, signal_std, stats


def split_dataset(dataset, train_ratio: float = 0.6, val_ratio: float = 0.2):
    """数据集划分（训练/验证/测试）。"""
    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len
    return random_split(dataset, [train_len, val_len, test_len])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class WaveUNetWithTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=15, padding=7), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=15, padding=7), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.encoder4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, padding=7), nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.transformer_input_proj = nn.Conv1d(128, 256, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.transformer_output_proj = nn.Conv1d(256, 128, kernel_size=1)

        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
            nn.Conv1d(128, 64, kernel_size=15, padding=7), nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
            nn.Conv1d(64, 32, kernel_size=15, padding=7), nn.ReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
            nn.Conv1d(32, 16, kernel_size=15, padding=7), nn.ReLU(),
        )
        self.decoder4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
            nn.Conv1d(16, 1, kernel_size=15, padding=7),
        )

    def forward(self, x, mask):
        input_len = x.shape[-1]
        residual = x

        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        x_trans = self.transformer_input_proj(x4).permute(0, 2, 1)
        x_trans = self.transformer(x_trans)
        x_trans = self.transformer_output_proj(x_trans.permute(0, 2, 1))

        x = self.decoder1(x_trans)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)

        if x.shape[-1] > input_len:
            x = x[:, :, :input_len]
        elif x.shape[-1] < input_len:
            x = F.pad(x, (0, input_len - x.shape[-1]))

        output = (
            x * (mask == 1).float() + residual * (mask == 0).float()
        ) * (mask != 2).float()

        return output


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def masked_loss(output, target, mask, lambda_mse=5.0, lambda_smooth=1.0, stability_penalty=1.0):
    """掩码损失函数：MSE + 平滑项 + 防塌缩标准差惩罚（仅作用于弱信号①）。"""
    loss = F.mse_loss(output, target, reduction="none")
    active = (mask == 1).float()
    core = (loss * active).sum() / (active.sum() + 1e-8)

    diff = output[:, :, 1:] - output[:, :, :-1]
    mask_diff = (mask[:, :, 1:] == 1) & (mask[:, :, :-1] == 1)
    smooth_penalty = (diff ** 2 * mask_diff.float()).sum() / (mask_diff.float().sum() + 1e-8)

    weak_output = output[mask == 1]
    std_penalty = 1.0 / (torch.std(weak_output) + 1e-4)

    total_loss = lambda_mse * core + lambda_smooth * smooth_penalty + stability_penalty * std_penalty
    return total_loss


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------
class ModelCheckpoint:
    """完整的模型保存和加载系统。"""

    def __init__(self, save_dir: str = "checkpoints"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        train_loss,
        val_loss,
        best_val_loss,
        model_config=None,
        training_config=None,
        checkpoint_name=None,
    ):
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_epoch_{epoch}_{timestamp}.pt"

        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "timestamp": datetime.now().isoformat(),
            "model_config": model_config or {},
            "training_config": training_config or {},
        }

        torch.save(checkpoint, checkpoint_path)

        metadata = {
            "checkpoint_name": checkpoint_name,
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "best_val_loss": float(best_val_loss),
            "timestamp": checkpoint["timestamp"],
            "model_config": model_config or {},
            "training_config": training_config or {},
        }

        metadata_path = os.path.join(
            self.save_dir, checkpoint_name.replace(".pt", "_metadata.json")
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ 检查点已保存: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"):
        """加载检查点以继续训练。

        同时兼容两类文件:
        1. 完整检查点 (``checkpoint_epoch_*.pt`` / ``final_checkpoint_*.pt``)：包含
           ``model_state_dict`` + ``optimizer_state_dict`` + ``scheduler_state_dict`` +
           ``epoch`` + ``best_val_loss`` 等，可完整恢复训练状态并接着 epoch 续训。
        2. 最佳模型 (``best_denoising_model*.pt``)：只含 ``model_state_dict`` / ``val_loss`` /
           ``timestamp``。此时优雅降级——只恢复模型权重，optimizer/scheduler 保持新建状态，
           epoch 从 0 计数，并以文件里的 ``val_loss`` 作为起始 best_val_loss。
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        print(f"📂 正在加载检查点: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 兼容直接保存 state_dict 的极简文件
        if "model_state_dict" not in checkpoint:
            model.load_state_dict(checkpoint)
            checkpoint = {"model_state_dict": checkpoint}
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        has_optimizer_state = "optimizer_state_dict" in checkpoint
        is_best_model_file = not has_optimizer_state and "epoch" not in checkpoint

        if optimizer and has_optimizer_state:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("✅ 优化器状态已加载")
        elif optimizer:
            print("ℹ️ 文件中无优化器状态 (疑似 best_model)，使用新建优化器继续训练")

        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("✅ 学习率调度器状态已加载")
        elif scheduler and is_best_model_file:
            print("ℹ️ 文件中无调度器状态 (疑似 best_model)，使用新建调度器继续训练")

        # best_model 文件没有 best_val_loss，用它自带的 val_loss 作为起始 best 更合理，
        # 否则起始 best=inf 会导致第一轮无脑覆盖最佳模型。
        fallback_best = checkpoint.get("val_loss", float("inf"))
        loaded_info = {
            "epoch": checkpoint.get("epoch", 0),
            "train_loss": checkpoint.get("train_loss", float("inf")),
            "val_loss": checkpoint.get("val_loss", float("inf")),
            "best_val_loss": checkpoint.get("best_val_loss", fallback_best),
            "timestamp": checkpoint.get("timestamp", "unknown"),
            "model_config": checkpoint.get("model_config", {}),
            "training_config": checkpoint.get("training_config", {}),
            "is_best_model_file": is_best_model_file,
        }

        print(f"✅ 检查点加载完成!")
        print(f"   - 类型: {'best_model (仅权重)' if is_best_model_file else '完整检查点'}")
        print(f"   - 轮次: {loaded_info['epoch']}")
        print(f"   - 验证损失: {loaded_info['val_loss']:.6f}")
        print(f"   - 最佳验证损失: {loaded_info['best_val_loss']:.6f}")
        print(f"   - 保存时间: {loaded_info['timestamp']}")

        return loaded_info

    def save_best_model(self, model, val_loss, model_name="best_model.pt"):
        model_path = os.path.join(self.save_dir, model_name)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "timestamp": datetime.now().isoformat(),
            },
            model_path,
        )
        print(f"🏆 最佳模型已保存: {model_path} (验证损失: {val_loss:.6f})")
        return model_path

    def list_checkpoints(self):
        checkpoints = []
        for file in os.listdir(self.save_dir):
            if file.endswith(".pt") and not file.startswith("best_"):
                metadata_file = file.replace(".pt", "_metadata.json")
                metadata_path = os.path.join(self.save_dir, metadata_file)

                if os.path.exists(metadata_path):
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    checkpoints.append(metadata)
                else:
                    checkpoints.append({"checkpoint_name": file, "epoch": "unknown"})

        checkpoints.sort(key=lambda x: x.get("epoch", 0))
        return checkpoints

    def cleanup_old_checkpoints(self, keep_last_n=5):
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= keep_last_n:
            print(
                f"当前检查点数量({len(checkpoints)})未超过保留数量({keep_last_n})，无需清理"
            )
            return

        to_remove = checkpoints[:-keep_last_n]
        for checkpoint in to_remove:
            checkpoint_name = checkpoint["checkpoint_name"]
            checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
            metadata_path = checkpoint_path.replace(".pt", "_metadata.json")

            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"🗑️ 已删除旧检查点: {checkpoint_name}")

            if os.path.exists(metadata_path):
                os.remove(metadata_path)

        print(f"✅ 清理完成，保留了最近的 {keep_last_n} 个检查点")


# ---------------------------------------------------------------------------
# Training loop with checkpoint support
# ---------------------------------------------------------------------------
def train_model_with_checkpoints(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=120,
    checkpoint_manager=None,
    resume_from_checkpoint=None,
    save_every_n_epochs=10,
    model_config=None,
    training_config=None,
    best_model_name="best_denoising_model.pt",
):
    start_epoch = 0
    best_val_loss = float("inf")
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True)

    if resume_from_checkpoint and checkpoint_manager:
        try:
            loaded_info = checkpoint_manager.load_checkpoint(
                resume_from_checkpoint, model, optimizer, scheduler, device
            )
            start_epoch = loaded_info["epoch"]
            best_val_loss = loaded_info["best_val_loss"]
            if loaded_info.get("is_best_model_file"):
                print(
                    f"🔄 从 best_model 权重继续训练 (epoch 从 0 计数)，"
                    f"起始最佳验证损失: {best_val_loss:.6f}"
                )
            else:
                print(f"🔄 从轮次 {start_epoch} 继续训练，当前最佳验证损失: {best_val_loss:.6f}")
        except Exception as e:
            print(f"❌ 加载检查点失败: {e}")
            print("🆕 从头开始训练")
            start_epoch = 0
            best_val_loss = float("inf")

    if training_config is None:
        training_config = {
            "num_epochs": num_epochs,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "optimizer": type(optimizer).__name__,
            "scheduler": type(scheduler).__name__,
            "device": str(device),
            "batch_size": train_loader.batch_size,
            "train_dataset_size": len(train_loader.dataset),
            "val_dataset_size": len(val_loader.dataset),
        }

    if model_config is None:
        model_config = {
            "model_type": type(model).__name__,
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

    print("🚀 开始训练:")
    print(f"   - 模型: {model_config['model_type']}")
    print(f"   - 总参数量: {model_config['total_params']:,}")
    print(f"   - 可训练参数: {model_config['trainable_params']:,}")
    print(f"   - 设备: {device}")
    print(f"   - 训练轮次: {start_epoch} -> {num_epochs}")
    print(f"   - 学习率: {training_config['learning_rate']}")

    train_loss = float("inf")
    val_loss = float("inf")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            white_data = batch["white_data"].to(device)
            white_signal = batch["white_signal"].to(device)

            mask = generate_mask(white_data).unsqueeze(1).to(device)

            x = white_data.unsqueeze(1)
            y = white_signal.unsqueeze(1)

            x_std, y_std, _ = standardize_batch(x.squeeze(1), y.squeeze(1), mask.squeeze(1))
            x_std = x_std.unsqueeze(1).to(device)
            y_std = y_std.unsqueeze(1).to(device)

            optimizer.zero_grad()
            output = model(x_std, mask)
            loss = masked_loss(output, y_std, mask)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        train_loss /= max(train_batches, 1)

        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                white_data = batch["white_data"].to(device)
                white_signal = batch["white_signal"].to(device)

                mask = generate_mask(white_data).unsqueeze(1).to(device)

                x = white_data.unsqueeze(1)
                y = white_signal.unsqueeze(1)

                x_std, y_std, _ = standardize_batch(x.squeeze(1), y.squeeze(1), mask.squeeze(1))
                x_std = x_std.unsqueeze(1).to(device)
                y_std = y_std.unsqueeze(1).to(device)

                output = model(x_std, mask)
                loss = masked_loss(output, y_std, mask)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= max(val_batches, 1)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        output_std = output.std().item() if "output" in locals() else 0.0

        print(
            f"{epoch + 1:3d}/{num_epochs}: Train Loss = {train_loss:.6f}, "
            f"Val Loss = {val_loss:.6f}, Output Std = {output_std:.4f}, LR = {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if checkpoint_manager:
                checkpoint_manager.save_best_model(model, val_loss, best_model_name)
            else:
                torch.save(model.state_dict(), best_model_name)
            print("  🏆 Best model saved!")

        if checkpoint_manager and (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, epoch + 1, train_loss, val_loss,
                best_val_loss, model_config, training_config,
            )

        if checkpoint_manager and (epoch + 1) % 50 == 0:
            checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)

    if checkpoint_manager:
        final_checkpoint = checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, num_epochs, train_loss, val_loss,
            best_val_loss, model_config, training_config,
            checkpoint_name=f"final_checkpoint_epoch_{num_epochs}.pt",
        )
        print(f"🎯 训练完成! 最终检查点: {final_checkpoint}")

    print(f"✅ 训练完成! 最佳验证损失: {best_val_loss:.6f}")
    return best_val_loss


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train WaveU-Net + Transformer denoising model from scratch.",
    )
    parser.add_argument(
        "--data-folder", type=str, default="./waveforms2_2sig",
        help="存放 waveforms_proc*.h5 的目录 (默认: ./waveforms2_2sig)",
    )
    parser.add_argument(
        "--target-snr", type=str, nargs="+", default=["200.00"],
        help='本轮训练使用的 SNR 列表，字符串形式，如 "200.00" "100.00" (默认: 200.00)',
    )
    parser.add_argument("--num-epochs", type=int, default=200, help="训练总轮次 (默认: 200)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size (默认: 32)")
    parser.add_argument(
        "--num-workers", type=int, default=24,
        help="DataLoader worker 数量 (默认: 24，HPC A100 节点的设置)",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="学习率 (默认: 1e-5)")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="weight decay (默认: 1e-6)")
    parser.add_argument(
        "--save-every-n-epochs", type=int, default=10,
        help="每隔多少轮保存一次检查点 (默认: 10)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="model_checkpoints",
        help="检查点保存目录 (默认: model_checkpoints)",
    )
    parser.add_argument(
        "--best-model-name", type=str, default="best_denoising_model.pt",
        help="最佳模型文件名 (默认: best_denoising_model.pt)",
    )
    parser.add_argument(
        "--resume-from-checkpoint", type=str, default=None,
        help=(
            "可选: 继续训练所使用的 .pt 文件，自动识别两类: "
            "(1) 完整检查点 checkpoint_epoch_*.pt → 恢复 权重+优化器+调度器+epoch, 接着 epoch 续训; "
            "(2) best_denoising_model*.pt → 仅恢复权重, 新建优化器/调度器, epoch 从 0 计数。"
            "与 --finetune-from 互斥。"
        ),
    )
    parser.add_argument(
        "--finetune-from", type=str, default=None,
        help=(
            "可选: 从指定的 .pt 文件 (best_denoising_model.pt 或任一 checkpoint) "
            "仅加载模型权重做微调，不加载优化器/调度器/epoch。epoch 计数从 0 开始，"
            "使用 --learning-rate 给定的较小学习率训练新数据。与 --resume-from-checkpoint 互斥。"
        ),
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.6, help="训练集比例 (默认: 0.6)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="验证集比例 (默认: 0.2)",
    )
    parser.add_argument("--seed", type=int, default=None, help="可选随机种子")
    parser.add_argument(
        "--no-pin-memory", action="store_true",
        help="禁用 pin_memory (调试或共享内存紧张时使用)",
    )
    return parser.parse_args()


def _load_weights_for_finetune(model, weights_path: str, device) -> None:
    """仅加载模型权重用于 fine-tune。

    支持两种文件:
    1. `best_denoising_model.pt`：``{"model_state_dict": ..., "val_loss": ..., "timestamp": ...}``
    2. 完整检查点 `checkpoint_epoch_*.pt`：包含 ``model_state_dict`` 以及 optimizer / scheduler /
       epoch / training_config 等其它键，本函数会**忽略**除 ``model_state_dict`` 之外的所有内容。
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    print(f"📂 正在为微调加载权重: {weights_path}")
    payload = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        if "val_loss" in payload:
            print(f"   - 预训练 val_loss: {payload['val_loss']:.6f}")
        if "epoch" in payload:
            print(f"   - 预训练 epoch: {payload['epoch']}")
        if "timestamp" in payload:
            print(f"   - 保存时间: {payload['timestamp']}")
    else:
        # 兼容直接保存 state_dict 的旧文件
        state_dict = payload

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"⚠️ 缺失的键: {missing}")
    if unexpected:
        print(f"⚠️ 多余的键: {unexpected}")
    print("✅ 模型权重已加载，准备开始 fine-tune (epoch 从 0 计数)")


def main():
    args = parse_args()

    if args.resume_from_checkpoint and args.finetune_from:
        raise ValueError(
            "--resume-from-checkpoint 与 --finetune-from 不能同时指定:"
            " 前者用于恢复完整训练状态, 后者只加载权重做微调"
        )

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print(f"加载数据集: {args.data_folder} (SNR={args.target_snr})")
    dataset = GWDataset(args.data_folder, args.target_snr)

    train_set, val_set, test_set = split_dataset(
        dataset, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
    )
    print(
        f"数据集划分: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}"
    )

    pin_memory = (not args.no_pin_memory) and (device.type == "cuda")
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )
    # 保留 test_loader 接口，方便后续按需调用，但训练阶段不使用
    _ = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )

    denoising_model = WaveUNetWithTransformer().to(device)

    if args.finetune_from:
        _load_weights_for_finetune(denoising_model, args.finetune_from, device)

    optimizer = torch.optim.Adam(
        denoising_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    checkpoint_manager = ModelCheckpoint(save_dir=args.checkpoint_dir)
    print("✅ 模型检查点管理器已初始化")

    training_config = {
        "snr": list(args.target_snr),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "save_every_n_epochs": args.save_every_n_epochs,
        "number of epochs": args.num_epochs,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "num_workers": args.num_workers,
    }
    if args.finetune_from:
        training_config.update({
            "fine_tuning": True,
            "finetune_from": args.finetune_from,
            "new_snr": list(args.target_snr),
        })

    best_loss = train_model_with_checkpoints(
        model=denoising_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        checkpoint_manager=checkpoint_manager,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_every_n_epochs=args.save_every_n_epochs,
        training_config=training_config,
        best_model_name=args.best_model_name,
    )

    print(f"✅ 训练完成! 最佳验证损失: {best_loss:.6f}")


if __name__ == "__main__":
    main()
