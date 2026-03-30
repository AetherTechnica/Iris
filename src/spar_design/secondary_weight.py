"""
secondary_weight.py — 二次構造（外皮・リブ）重量分布
=====================================================
仕様書 Section 6.4 に対応。全て mm / kg 単位。

【コード長分布】
  楕円分布: c(y) = c_root × √(1 - (y/span)²)

【外皮重量】
  上下面を考慮: A_skin = c(y) × dy × 2 [mm²] → 換算して kg

【リブ重量】
  rib_pitch ごとに 1 個，RIB_WEIGHT [kg/個]
"""

from __future__ import annotations
import numpy as np

try:
    from .config import RIB_PITCH, RIB_WEIGHT, GRID_SIZE
except ImportError:
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..'))
    from src.spar_design.config import RIB_PITCH, RIB_WEIGHT, GRID_SIZE


def chord_elliptic(y_mm: np.ndarray, span_mm: float, root_chord_mm: float) -> np.ndarray:
    """楕円コード長分布 [mm]。"""
    eta = y_mm / span_mm
    return root_chord_mm * np.sqrt(np.maximum(0.0, 1.0 - eta**2))


def secondary_weight_distribution(
    y_mm: np.ndarray,
    span_mm: float,
    root_chord_mm: float,
    rho_skin_kg_m2: float,
) -> np.ndarray:
    """
    スパン方向の二次構造分布重量 [kg/mm]（各 GRID_SIZE 区間あたり）。

    Args:
        y_mm          : スパン方向位置配列 [mm]
        span_mm       : 片翼スパン [mm]
        root_chord_mm : 翼根コード長 [mm]
        rho_skin_kg_m2: 外皮面密度 [kg/m²]

    Returns:
        w_dist [kg/mm] — y_mm と同じ shape の配列
    """
    dy = GRID_SIZE  # mm

    # コード長 [mm]
    c = chord_elliptic(y_mm, span_mm, root_chord_mm)

    # 外皮重量（上下面，GRID_SIZE 区間）
    # A_skin [m²] = c[mm] × dy[mm] × 2 × 1e-6
    rho_skin_g_mm2 = rho_skin_kg_m2 * 1e-6  # kg/m² → kg/mm²
    w_skin = rho_skin_g_mm2 * c * dy * 2.0  # [kg]（区間あたり）

    # リブ重量（RIB_PITCH ごとに 1 個）
    # y が RIB_PITCH の倍数に最も近い区間に割り当て
    rib_mod = np.abs(y_mm % RIB_PITCH)
    has_rib = (rib_mod < dy / 2) | (rib_mod > RIB_PITCH - dy / 2)
    w_rib = np.where(has_rib, RIB_WEIGHT, 0.0)  # [kg]

    # kg/mm に変換（区間重量 ÷ 区間長）
    w_dist = (w_skin + w_rib) / dy  # [kg/mm]

    return w_dist


def total_secondary_weight(
    y_mm: np.ndarray,
    span_mm: float,
    root_chord_mm: float,
    rho_skin_kg_m2: float,
) -> float:
    """二次構造の片翼総重量 [kg]。"""
    w_dist = secondary_weight_distribution(y_mm, span_mm, root_chord_mm, rho_skin_kg_m2)
    return float(np.trapz(w_dist, y_mm))


if __name__ == "__main__":
    span = 10000.0
    y = np.arange(0, span + GRID_SIZE, GRID_SIZE)
    w = secondary_weight_distribution(y, span, 300.0, 0.05)
    total = total_secondary_weight(y, span, 300.0, 0.05)
    print(f"二次構造重量（片翼）: {total*1000:.1f} g")
    print(f"最大分布荷重: {w.max()*1000:.4f} g/mm  @ y=0")
