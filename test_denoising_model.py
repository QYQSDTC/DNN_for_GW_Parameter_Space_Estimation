"""Evaluate the trained ``best_denoising_model.pt`` on the test split.

本脚本与 ``train_from_scratch.py`` 配套使用：复用其中的 ``GWDataset``、
``WaveUNetWithTransformer``、掩码 / 标准化工具、``masked_loss`` 与
``ModelCheckpoint`` 等组件，对训练好的最佳去噪模型 (默认文件名
``best_denoising_model.pt``) 在测试集上做端到端评估。

主要功能：
    1. 加载 ``best_denoising_model.pt`` 中的权重。
    2. 复现 ``train_from_scratch.py`` 中相同的数据划分 (使用同一个 ``--seed``
       即可得到与训练时一致的 train/val/test 切分)。
    3. 在测试集上计算
        - masked loss (与训练阶段一致, 用于和训练日志直接对照)
        - 仅弱信号 ① 区域的 MSE / MAE
        - 输入 vs 信号 的 MSE 作为基线 (denoise 之前)
        - 输出弱信号 ① 区域的标准差均值 (检查是否塌缩)
        - 简单 SNR 改善 (Δ SNR_dB)
    4. (可选) 用 ``test_model_and_save_labels_batched`` 把去噪结果按 batch
       保存为 ``.pt``，供下游预测网络使用。
    5. (可选) 用 ``visualize_denoising_subplots`` 将若干样本的输入 / 输出 /
       真值波形对比图保存到指定目录。

使用示例：

    python test_denoising_model.py \\
        --model-path ./best_denoising_model.pt \\
        --data-folder ./waveforms2_2sig \\
        --target-snr 200.00 \\
        --batch-size 32 \\
        --num-workers 8 \\
        --seed 42 \\
        --save-denoised-dir ./pt_chunks \\
        --visualize-indices 0 5 21 \\
        --plot-dir ./denoising_plots
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # 后台运行时不依赖 GUI
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from train_from_scratch import (
    GWDataset,
    WaveUNetWithTransformer,
    generate_mask,
    masked_loss,
    split_dataset,
    standardize_batch,
)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_best_model(model: torch.nn.Module, model_path: str, device: torch.device):
    """加载 ``best_denoising_model.pt``。

    兼容两种保存格式：
        - ``{"model_state_dict": ..., "val_loss": ..., "timestamp": ...}``
          (``ModelCheckpoint.save_best_model`` 的格式)
        - 纯 ``state_dict`` (直接 ``torch.save(model.state_dict(), ...)``)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"📂 正在加载最佳模型: {model_path}")
    payload = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        info = {
            "val_loss": payload.get("val_loss"),
            "timestamp": payload.get("timestamp"),
        }
    else:
        state_dict = payload
        info = {"val_loss": None, "timestamp": None}

    model.load_state_dict(state_dict)
    print("✅ 模型加载完成")
    if info["val_loss"] is not None:
        print(f"   - 训练记录的最佳验证损失: {info['val_loss']:.6f}")
    if info["timestamp"] is not None:
        print(f"   - 保存时间: {info['timestamp']}")
    return info


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_on_loader(model, loader, device, verbose_every: int = 20):
    """遍历 loader, 计算 masked loss + 多种诊断指标。"""
    model.eval()

    sum_masked_loss = 0.0
    sum_weak_mse = 0.0
    sum_weak_mae = 0.0
    sum_weak_count = 0  # 弱信号 ① 区域的总样点数
    sum_baseline_mse = 0.0  # 输入 vs 真信号 (在弱信号 ① 区域)
    sum_input_power = 0.0   # 输入弱信号区域的能量
    sum_resid_power = 0.0   # 残差能量 (output - target)
    sum_output_std = 0.0
    n_batches = 0
    n_samples = 0

    for batch_idx, batch in enumerate(loader):
        white_data = batch["white_data"].to(device)
        white_signal = batch["white_signal"].to(device)
        bsz = white_data.shape[0]

        mask = generate_mask(white_data).unsqueeze(1).to(device)

        x = white_data.unsqueeze(1)
        y = white_signal.unsqueeze(1)

        x_std, y_std, _ = standardize_batch(
            x.squeeze(1), y.squeeze(1), mask.squeeze(1)
        )
        x_std = x_std.unsqueeze(1).to(device)
        y_std = y_std.unsqueeze(1).to(device)

        output = model(x_std, mask)
        loss = masked_loss(output, y_std, mask)

        weak = (mask == 1).float()
        weak_count = weak.sum().item()

        if weak_count > 0:
            diff = (output - y_std) * weak
            weak_mse = (diff.pow(2).sum() / weak_count).item()
            weak_mae = (diff.abs().sum() / weak_count).item()

            base_diff = (x_std - y_std) * weak
            baseline_mse = (base_diff.pow(2).sum() / weak_count).item()

            input_power = ((x_std * weak).pow(2).sum() / weak_count).item()
            resid_power = (diff.pow(2).sum() / weak_count).item()

            weak_output = output[mask == 1]
            output_std = weak_output.std().item() if weak_output.numel() > 0 else 0.0
        else:
            weak_mse = weak_mae = baseline_mse = input_power = resid_power = output_std = 0.0

        sum_masked_loss += loss.item()
        sum_weak_mse += weak_mse * max(weak_count, 1)
        sum_weak_mae += weak_mae * max(weak_count, 1)
        sum_baseline_mse += baseline_mse * max(weak_count, 1)
        sum_input_power += input_power * max(weak_count, 1)
        sum_resid_power += resid_power * max(weak_count, 1)
        sum_weak_count += max(weak_count, 1)
        sum_output_std += output_std
        n_batches += 1
        n_samples += bsz

        if verbose_every and (batch_idx + 1) % verbose_every == 0:
            print(
                f"  [batch {batch_idx + 1}/{len(loader)}] "
                f"masked_loss={loss.item():.6f}, weak_mse={weak_mse:.6f}, "
                f"output_std={output_std:.4f}"
            )

    if n_batches == 0:
        raise RuntimeError("测试 DataLoader 为空，无法评估")

    avg_masked_loss = sum_masked_loss / n_batches
    avg_weak_mse = sum_weak_mse / sum_weak_count
    avg_weak_mae = sum_weak_mae / sum_weak_count
    avg_baseline_mse = sum_baseline_mse / sum_weak_count
    avg_input_power = sum_input_power / sum_weak_count
    avg_resid_power = sum_resid_power / sum_weak_count
    avg_output_std = sum_output_std / n_batches

    eps = 1e-12
    snr_in_db = 10.0 * np.log10(max(avg_input_power, eps) / max(avg_baseline_mse, eps))
    snr_out_db = 10.0 * np.log10(max(avg_input_power, eps) / max(avg_resid_power, eps))
    snr_improvement_db = snr_out_db - snr_in_db
    mse_improvement_ratio = avg_baseline_mse / max(avg_weak_mse, eps)

    metrics = {
        "n_batches": n_batches,
        "n_samples": n_samples,
        "avg_masked_loss": avg_masked_loss,
        "avg_weak_mse": avg_weak_mse,
        "avg_weak_mae": avg_weak_mae,
        "avg_baseline_mse_input_vs_signal": avg_baseline_mse,
        "avg_output_weak_std": avg_output_std,
        "snr_input_db": snr_in_db,
        "snr_output_db": snr_out_db,
        "snr_improvement_db": snr_improvement_db,
        "mse_improvement_ratio": mse_improvement_ratio,
    }
    return metrics


# ---------------------------------------------------------------------------
# Save denoised batches (与 notebook 中 test_model_and_save_labels_batched 等价)
# ---------------------------------------------------------------------------
@torch.no_grad()
def save_denoised_batches(
    model,
    loader,
    device,
    output_dir: str,
    base_name: str = "denoised_batch",
    target_names=("mc_true", "phis_true", "thetas_true"),
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    batch_count = 0

    for batch in loader:
        white_data = batch["white_data"].to(device)
        mask = generate_mask(white_data).unsqueeze(1).to(device)
        x = white_data.unsqueeze(1)

        x_std, _, _ = standardize_batch(
            x.squeeze(1), x.squeeze(1), mask.squeeze(1)
        )
        x_std = x_std.unsqueeze(1).to(device)

        output = model(x_std, mask).cpu()

        targets = torch.stack(
            [
                torch.as_tensor(batch[k], dtype=torch.float32).view(-1)
                for k in target_names
            ],
            dim=1,
        )

        file_path = os.path.join(output_dir, f"{base_name}_{batch_count:04d}.pt")
        torch.save(
            {
                "denoised": output,
                "targets": targets,
                "target_names": list(target_names),
            },
            file_path,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"[Saved] {file_path}  ←  {output.shape[0]} samples")
        batch_count += 1

    print(f"[Done] 共保存 {batch_count} 个批次文件于: {output_dir}")
    return batch_count


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def selective_inverse_transform(signal, mask, stats, amplification: float = 1.0):
    signal = signal.copy()
    signal = signal * stats["iqr"] + stats["median"]
    signal[mask == 1] /= amplification
    return signal


@torch.no_grad()
def visualize_denoising(
    model,
    loader,
    device,
    sample_indices,
    save_dir: str = "denoising_plots",
):
    """从 loader 的第一个 batch 中挑选若干样本作对比图。"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    batch = next(iter(loader))
    white_data = batch["white_data"].to(device)
    white_signal = batch["white_signal"].to(device)
    bsz = white_data.shape[0]

    mask = generate_mask(white_data).unsqueeze(1).to(device)

    x = white_data.unsqueeze(1)
    y = white_signal.unsqueeze(1)

    x_std, y_std, stats = standardize_batch(
        x.squeeze(1), y.squeeze(1), mask.squeeze(1)
    )
    x_std = x_std.unsqueeze(1).to(device)
    y_std = y_std.unsqueeze(1).to(device)

    output = model(x_std, mask).squeeze(1).cpu().numpy()
    saved_paths = []

    for sample_index in sample_indices:
        if sample_index >= bsz:
            print(f"⚠️ 样本索引 {sample_index} 超过 batch_size={bsz}, 跳过")
            continue

        input_signal = x[sample_index].squeeze().cpu().numpy()
        denoised_signal = output[sample_index]
        clean_signal = y[sample_index].squeeze().cpu().numpy()
        signal_mask = mask[sample_index].squeeze().cpu().numpy()

        input_signal = selective_inverse_transform(input_signal, signal_mask, stats, 1.0)
        denoised_signal = selective_inverse_transform(denoised_signal, signal_mask, stats, 1.0)
        clean_signal = selective_inverse_transform(clean_signal, signal_mask, stats, 1.0)

        fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
        axs[0].plot(input_signal, color="orange")
        axs[0].set_title(f"Noisy Input (sample {sample_index})", fontsize=14)
        axs[1].plot(denoised_signal, color="green")
        axs[1].set_title("Denoised Output", fontsize=14)
        axs[2].plot(clean_signal, color="blue")
        axs[2].set_title("Ground Truth Signal", fontsize=14)
        for ax in axs:
            ax.grid(True)
        plt.tight_layout()

        save_path = os.path.join(
            save_dir, f"denoising_comparison_sample_{sample_index}.png"
        )
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ 去噪对比图已保存: {save_path}")
        saved_paths.append(save_path)

    return saved_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate best_denoising_model.pt on the test split.",
    )
    parser.add_argument(
        "--model-path", type=str, default="./best_denoising_model.pt",
        help="best_denoising_model.pt 路径 (默认: ./best_denoising_model.pt)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="(可选) 与 ModelCheckpoint 配合：若提供, 则从 "
             "<checkpoint-dir>/<best-model-name> 加载, 覆盖 --model-path",
    )
    parser.add_argument(
        "--best-model-name", type=str, default="best_denoising_model.pt",
        help="配合 --checkpoint-dir 使用的最佳模型文件名",
    )
    parser.add_argument(
        "--data-folder", type=str, default="./waveforms2_2sig",
        help="存放 waveforms_proc*.h5 的目录",
    )
    parser.add_argument(
        "--target-snr", type=str, nargs="+", default=["200.00"],
        help='测试使用的 SNR 列表，字符串形式，如 "200.00" "100.00"',
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="DataLoader worker 数量 (测试时通常无需太多)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.6,
        help="训练集比例 (默认与训练脚本一致)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="验证集比例 (默认与训练脚本一致)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="数据划分随机种子. 务必与训练时一致以避免泄漏",
    )
    parser.add_argument(
        "--no-pin-memory", action="store_true",
        help="禁用 pin_memory (CPU 测试或共享内存紧张时使用)",
    )
    parser.add_argument(
        "--use-full-dataset", action="store_true",
        help="忽略划分, 直接对整个数据集 (按 SNR 过滤后) 进行评估",
    )
    parser.add_argument(
        "--use-val-set", action="store_true",
        help="使用验证集而非测试集进行评估 (默认: False, 即测试集)",
    )

    parser.add_argument(
        "--save-denoised-dir", type=str, default=None,
        help="若指定, 把去噪结果按 batch 保存到该目录 (兼容 notebook 行为)",
    )
    parser.add_argument(
        "--visualize-indices", type=int, nargs="*", default=None,
        help="测试集第一个 batch 内要可视化的样本索引列表",
    )
    parser.add_argument(
        "--plot-dir", type=str, default="./denoising_plots",
        help="可视化保存目录",
    )
    parser.add_argument(
        "--metrics-json", type=str, default=None,
        help="若指定, 把指标 dict 保存为 JSON",
    )
    parser.add_argument(
        "--verbose-every", type=int, default=20,
        help="每多少个 batch 打印一次进度 (设为 0 关闭)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if args.checkpoint_dir is not None:
        model_path = os.path.join(args.checkpoint_dir, args.best_model_name)
    else:
        model_path = args.model_path

    print(f"加载数据集: {args.data_folder} (SNR={args.target_snr})")
    dataset = GWDataset(args.data_folder, args.target_snr)

    if args.use_full_dataset:
        eval_set = dataset
        split_name = "full"
        print(f"评估范围: 全量数据 ({len(eval_set)} 个样本)")
    else:
        if args.seed is not None:
            generator = torch.Generator().manual_seed(args.seed)
            total_len = len(dataset)
            train_len = int(total_len * args.train_ratio)
            val_len = int(total_len * args.val_ratio)
            test_len = total_len - train_len - val_len
            train_set, val_set, test_set = random_split(
                dataset, [train_len, val_len, test_len], generator=generator,
            )
        else:
            train_set, val_set, test_set = split_dataset(
                dataset, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
            )

        if args.use_val_set:
            eval_set = val_set
            split_name = "val"
        else:
            eval_set = test_set
            split_name = "test"

        print(
            f"数据集划分: train={len(train_set)}, val={len(val_set)}, "
            f"test={len(test_set)} (使用 {split_name} 集评估)"
        )

    pin_memory = (not args.no_pin_memory) and (device.type == "cuda")
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    denoising_model = WaveUNetWithTransformer().to(device)
    load_info = load_best_model(denoising_model, model_path, device)

    print(
        f"\n🔍 在 {split_name} 集上评估 "
        f"({len(eval_set)} 个样本, {len(eval_loader)} 个 batch)..."
    )
    metrics = evaluate_on_loader(
        denoising_model, eval_loader, device,
        verbose_every=args.verbose_every,
    )

    print("\n=========== 评估结果 ===========")
    print(f"  样本总数:                          {metrics['n_samples']}")
    print(f"  Batch 总数:                        {metrics['n_batches']}")
    print(f"  平均 masked loss (与训练同口径):   {metrics['avg_masked_loss']:.6f}")
    print(f"  弱信号区域 MSE:                    {metrics['avg_weak_mse']:.6e}")
    print(f"  弱信号区域 MAE:                    {metrics['avg_weak_mae']:.6e}")
    print(f"  基线 MSE (输入 vs 信号, 弱区):     {metrics['avg_baseline_mse_input_vs_signal']:.6e}")
    print(f"  MSE 提升倍数 (基线 / 模型):        {metrics['mse_improvement_ratio']:.3f}x")
    print(f"  弱信号区域输出 std (越大越不塌缩): {metrics['avg_output_weak_std']:.4f}")
    print(f"  SNR_input  (dB):                   {metrics['snr_input_db']:.3f}")
    print(f"  SNR_output (dB):                   {metrics['snr_output_db']:.3f}")
    print(f"  ΔSNR (dB):                         {metrics['snr_improvement_db']:.3f}")
    print("================================\n")

    if args.metrics_json:
        record = {
            "model_path": os.path.abspath(model_path),
            "data_folder": os.path.abspath(args.data_folder),
            "target_snr": list(args.target_snr),
            "split": split_name,
            "seed": args.seed,
            "loaded_val_loss": load_info.get("val_loss"),
            "loaded_timestamp": load_info.get("timestamp"),
            "evaluated_at": datetime.now().isoformat(),
            "metrics": metrics,
        }
        os.makedirs(os.path.dirname(args.metrics_json) or ".", exist_ok=True)
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"📊 指标已保存: {args.metrics_json}")

    if args.visualize_indices:
        print("\n🖼️ 生成对比图...")
        visualize_denoising(
            denoising_model, eval_loader, device,
            sample_indices=args.visualize_indices,
            save_dir=args.plot_dir,
        )

    if args.save_denoised_dir:
        print("\n💾 保存去噪 batch 文件...")
        save_denoised_batches(
            denoising_model, eval_loader, device,
            output_dir=args.save_denoised_dir,
        )

    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()
