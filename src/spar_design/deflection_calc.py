"""
deflection_calc.py — たわみ・曲げモーメント計算（mm 単位）
==========================================================
仕様書 Section 6.2 に対応。全て mm / N·mm² / N 単位。

【曲げモーメント（翼端から積分）】
  V(y) = ∫_y^span L_net(s) ds    （翼端 V=0 の境界条件）
  M(y) = ∫_y^span V(s) ds        （翼端 M=0 の境界条件）
  → 翼端から翼根方向にトラペゾイド積分

【たわみ（翼根から積分）】
  κ(y) = M(y) / EI(y)            [1/mm]
  θ(y) = ∫_0^y κ(s) ds          （翼根 θ=0）
  δ(y) = ∫_0^y θ(s) ds          （翼根 δ=0）
  → scipy.integrate.cumulative_trapezoid を使用
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import cumulative_trapezoid


def bending_moment(
    L_net_N_per_mm: np.ndarray,
    y_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    正味荷重分布から剪断力・曲げモーメント分布を計算する。

    翼端 (y[-1]) を固定端として，翼根方向に積分する。

    Args:
        L_net_N_per_mm: 正味荷重分布 [N/mm]（y_mm と同 shape）
        y_mm           : スパン方向位置 [mm]（昇順）

    Returns:
        (shear [N], moment [N·mm])  — y_mm と同 shape
    """
    n = len(y_mm)
    shear  = np.zeros(n)
    moment = np.zeros(n)

    # 翼端 (i = n-1) から翼根 (i = 0) へ台形則で逐次積分
    for i in range(n - 2, -1, -1):
        dy = y_mm[i + 1] - y_mm[i]
        shear[i]  = shear[i + 1]  + (L_net_N_per_mm[i] + L_net_N_per_mm[i + 1]) / 2.0 * dy
        moment[i] = moment[i + 1] + (shear[i]           + shear[i + 1])           / 2.0 * dy

    return shear, moment


def compute_deflection(
    M_Nmm: np.ndarray,
    EI_Nmm2: np.ndarray,
    y_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    曲げモーメントと剛性分布からたわみ分布を計算する。

    境界条件: δ(0) = 0, θ(0) = 0（翼根固定）

    Args:
        M_Nmm  : 曲げモーメント分布 [N·mm]
        EI_Nmm2: 曲げ剛性分布 [N·mm²]
        y_mm   : スパン方向位置 [mm]（昇順）

    Returns:
        (deflection [mm], slope [rad])
    """
    # 曲率 κ = M / EI  [1/mm]
    curvature = M_Nmm / (EI_Nmm2 + 1e-12)

    # 一回積分 → 傾き θ [rad]（翼根 θ=0）
    slope = cumulative_trapezoid(curvature, y_mm, initial=0.0)

    # 二回積分 → たわみ δ [mm]（翼根 δ=0）
    deflection = cumulative_trapezoid(slope, y_mm, initial=0.0)

    return deflection, slope


if __name__ == "__main__":
    # 動作テスト: 均一荷重 + 均一剛性の片持ち梁（解析解と比較）
    span   = 10000.0   # mm
    q      = 1e-3      # N/mm（均一荷重）
    EI_val = 1e12      # N·mm²

    y = np.linspace(0, span, 201)
    L = np.full_like(y, q)
    _, M = bending_moment(L, y)
    delta, _ = compute_deflection(M, np.full_like(y, EI_val), y)

    # 解析解: δ_tip = qL⁴ / (8EI)
    delta_theory = q * span**4 / (8 * EI_val)
    print(f"数値解  δ_tip = {delta[-1]:.3f} mm")
    print(f"解析解  δ_tip = {delta_theory:.3f} mm")
    print(f"誤差   = {abs(delta[-1] - delta_theory)/delta_theory*100:.2f} %")
