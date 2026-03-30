"""
stiffness_optimizer.py — たわみ制約下の最適剛性分布算出
=========================================================
仕様書 Section 6.3 に対応。

【問題定式化】
  minimize:   Σ EI(y)        （EI 総和の最小化 ≈ 重量の近似）
  subject to: δ_tip ≤ δ_max （翼端たわみ制約）
              EI_min ≤ EI(y) ≤ EI_max

【最適化変数】
  x = log10(EI)（EI を対数スケールで扱うことで数値的安定性を向上）

【アルゴリズム】
  scipy.optimize.minimize (SLSQP)
  初期値: たわみ = δ_max となる均一 EI を計算して設定
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

try:
    from .deflection_calc import compute_deflection
    from .layup_optimizer import LayupOptimizer
    from .mandrel import MANDREL_LIST, Mandrel
    from .config import GRID_SIZE
except ImportError:
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..'))
    from src.spar_design.deflection_calc import compute_deflection
    from src.spar_design.layup_optimizer import LayupOptimizer
    from src.spar_design.mandrel import MANDREL_LIST, Mandrel
    from src.spar_design.config import GRID_SIZE


def _get_global_EI_bounds(
    mandrel_list: list[Mandrel],
    opt: LayupOptimizer,
) -> tuple[float, float]:
    """全マンドレルにわたる実現可能 EI の下限・上限 [N·mm²] を返す。"""
    ei_mins, ei_maxs = [], []
    for m in mandrel_list:
        lo, hi = opt.get_feasible_EI_range(m.diameter)
        ei_mins.append(lo)
        ei_maxs.append(hi)
    return min(ei_mins), max(ei_maxs)


def _initial_uniform_EI(
    M_Nmm: np.ndarray,
    y_mm: np.ndarray,
    delta_max_mm: float,
) -> float:
    """
    翼端たわみがちょうど δ_max になる均一 EI を解析的に推定する。

    均一 EI の場合:
      δ_tip = (1/EI) × ∫∫ M dy dy
    より EI = (∫∫ M dy dy) / δ_max
    """
    from scipy.integrate import cumulative_trapezoid
    theta_unnorm = cumulative_trapezoid(M_Nmm, y_mm, initial=0.0)
    delta_unnorm = cumulative_trapezoid(theta_unnorm, y_mm, initial=0.0)
    integral = float(delta_unnorm[-1])
    if integral <= 0 or delta_max_mm <= 0:
        return 1e10
    return integral / delta_max_mm


def optimize_stiffness(
    M_Nmm: np.ndarray,
    y_mm: np.ndarray,
    delta_max_mm: float,
    mandrel_list: list[Mandrel] | None = None,
    opt: LayupOptimizer | None = None,
    maxiter: int = 150,
    verbose: bool = False,
) -> np.ndarray:
    """
    たわみ制約を満たす最軽量剛性分布を SLSQP で求める。

    Args:
        M_Nmm       : 曲げモーメント分布 [N·mm]（y_mm と同 shape）
        y_mm        : スパン方向位置 [mm]
        delta_max_mm: 翼端たわみ許容値 [mm]
        mandrel_list: 利用可能なマンドレル（None → MANDREL_LIST）
        opt         : LayupOptimizer（None → 新規作成）
        maxiter     : SLSQP の最大反復回数
        verbose     : True で収束過程を表示

    Returns:
        EI_opt [N·mm²]（y_mm と同 shape の配列）
    """
    if mandrel_list is None:
        mandrel_list = MANDREL_LIST
    if opt is None:
        opt = LayupOptimizer()

    # EI の実現可能範囲（log スケール）
    EI_lo, EI_hi = _get_global_EI_bounds(mandrel_list, opt)
    log_lo = np.log10(max(EI_lo, 1.0))
    log_hi = np.log10(EI_hi)

    # 初期値: δ_tip = δ_max になる均一 EI
    EI_init = _initial_uniform_EI(M_Nmm, y_mm, delta_max_mm)
    EI_init = float(np.clip(EI_init, EI_lo, EI_hi))
    log_EI_init = np.full(len(y_mm), np.log10(EI_init))

    # ---- 目的関数（EI 総和の最小化）----------------------------
    def objective(log_ei: np.ndarray) -> float:
        return float(np.sum(10.0**log_ei))

    def obj_grad(log_ei: np.ndarray) -> np.ndarray:
        ei = 10.0**log_ei
        return ei * np.log(10.0)   # d(sum(EI))/d(log_ei_i) = EI_i * ln10

    # ---- 制約関数（翼端たわみ ≤ δ_max）------------------------
    def constraint_deflection(log_ei: np.ndarray) -> float:
        EI = 10.0**log_ei
        delta, _ = compute_deflection(M_Nmm, EI, y_mm)
        return float(delta_max_mm - delta[-1])   # ≥ 0 で制約満足

    bounds = [(log_lo, log_hi)] * len(y_mm)

    result = minimize(
        objective,
        x0          = log_EI_init,
        jac         = obj_grad,
        method      = "SLSQP",
        bounds      = bounds,
        constraints = {"type": "ineq", "fun": constraint_deflection},
        options     = {"maxiter": maxiter, "ftol": 1e-6, "disp": verbose},
    )

    if verbose:
        status = "収束" if result.success else f"未収束({result.message})"
        print(f"  SLSQP: {status}  ({result.nit} iter)")

    # 収束しない場合も物理的範囲内にクリップして返す
    log_EI_opt = np.clip(result.x, log_lo, log_hi)

    # たわみ制約が満たされているか確認し，不足時はスケールアップ
    EI_opt = 10.0**log_EI_opt
    delta_final, _ = compute_deflection(M_Nmm, EI_opt, y_mm)
    if delta_final[-1] > delta_max_mm * 1.01:
        scale = delta_final[-1] / delta_max_mm
        EI_opt = np.clip(EI_opt * scale, EI_lo, EI_hi)
        if verbose:
            print(f"  ⚠ たわみ超過 → EI を {scale:.3f} 倍にスケール")

    return EI_opt


if __name__ == "__main__":
    from src.spar_design.load_calculator import LoadCalculator

    print("=" * 60)
    print("  StiffnessOptimizer 動作テスト")
    print("=" * 60)

    calc = LoadCalculator(
        span_mm        = 10000.0,
        root_chord_mm  = 300.0,
        rho_skin_kg_m2 = 0.05,
        v_ms           = 7.5,
        rho_air        = 1.154,
    )
    M, _ = calc.moment_distribution_Nmm(W_total_kg=82.0, beta=0.9)
    y    = calc.y_mm

    EI_opt = optimize_stiffness(M, y, delta_max_mm=2000.0, verbose=True)

    from src.spar_design.deflection_calc import compute_deflection
    delta, _ = compute_deflection(M, EI_opt, y)

    print(f"\n翼根 EI  : {EI_opt[0]:.3e} N·mm²")
    print(f"翼端 EI  : {EI_opt[-1]:.3e} N·mm²")
    print(f"翼端たわみ: {delta[-1]:.1f} mm  (制約: 2000 mm)")
