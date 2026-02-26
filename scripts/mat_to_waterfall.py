#!/usr/bin/env python3
"""读取 MATLAB .mat 中的 DAS 数据并绘制时域信号波形图（幅值-时间）。

数据约定：
1. 第 1 行：相位差（phase difference）
2. 第 2 行：强度差（differential intensity）
3. 采样率：10 kHz（默认值，可通过参数覆盖）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DEFAULT_FS = 10_000.0


def _pick_signal_matrix(mat_dict: dict) -> np.ndarray:
    """从 .mat 结构中选择最可能的数据矩阵（二维数值数组）。"""
    candidates: list[tuple[str, np.ndarray]] = []
    for key, value in mat_dict.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and value.ndim == 2:
            candidates.append((key, value))

    if not candidates:
        raise ValueError("在 .mat 文件中未找到二维数值数组，请检查数据结构。")

    # 优先选择维度中包含 2 的矩阵（符合“相位差+强度差”双通道结构）
    for _, arr in candidates:
        if 2 in arr.shape:
            return arr

    # 否则回退到元素最多的二维数组
    return max(candidates, key=lambda item: item[1].size)[1]


def _extract_phase_and_intensity(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """提取相位差和强度差一维时域信号。"""
    if data.shape[0] >= 2:
        phase = data[0, :]
        intensity = data[1, :]
    elif data.shape[1] >= 2:
        # 若数据按列存储（N x 2），自动适配
        phase = data[:, 0]
        intensity = data[:, 1]
    else:
        raise ValueError(
            f"数据形状为 {data.shape}，无法提取前两行/前两列作为相位差和强度差。"
        )

    return np.asarray(phase).ravel(), np.asarray(intensity).ravel()


def plot_time_series(
    signal: np.ndarray,
    fs: float,
    title: str,
    out_file: Path,
    y_label: str = "Amplitude",
) -> None:
    """绘制单通道时域波形图（幅值-时间）。"""
    if signal.size == 0:
        raise ValueError("输入信号为空，无法绘图。")

    t = np.arange(signal.size, dtype=float) / fs

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.plot(t, signal, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="读取 .mat DAS 数据并生成时域波形图")
    parser.add_argument("mat_file", type=Path, help="输入 .mat 文件路径")
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=DEFAULT_FS,
        help=f"采样率 Hz（默认 {DEFAULT_FS:g}）",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("timeseries"),
        help="输出文件名前缀（默认 timeseries）",
    )
    args = parser.parse_args()

    mat = loadmat(args.mat_file)
    raw = _pick_signal_matrix(mat)
    phase, intensity = _extract_phase_and_intensity(raw)

    out_phase = Path(f"{args.output_prefix}_phase.png")
    out_intensity = Path(f"{args.output_prefix}_intensity.png")

    plot_time_series(phase, args.sampling_rate, "Phase Difference (Time Domain)", out_phase)
    plot_time_series(
        intensity,
        args.sampling_rate,
        "Differential Intensity (Time Domain)",
        out_intensity,
    )

    print(f"已生成: {out_phase}")
    print(f"已生成: {out_intensity}")


if __name__ == "__main__":
    main()
