"""
convergence_loop.py — 重量収束ループ（Picard反復）
====================================================
仕様書 Section 7 に対応。

【アルゴリズム概要】
  1. 初期桁重量 w_spar を推定値で与える
  2. LoadCalculator で荷重・曲げモーメントを計算
  3. StiffnessOptimizer で最適 EI 分布を算出
  4. MandrelDP で最適マンドレル配置を決定
  5. DP 結果から新しい桁重量分布を構築
  6. 重量変化が CONV_TOL 以下になるまで繰り返し

【緩和（Relaxation）】
  w_new = α × w_dp + (1−α) × w_old
  発散防止のため α = RELAX_ALPHA（デフォルト 0.5）
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

try:
    from .load_calculator import LoadCalculator
    from .stiffness_optimizer import optimize_stiffness
    from .mandrel_dp import mandrel_selection_dp, DPResult
    from .mandrel import MANDREL_LIST, Mandrel
    from .layup_optimizer import LayupOptimizer
    from .config import (
        GRID_SIZE, MAX_ITER, CONV_TOL, RELAX_ALPHA,
        DEFAULT_SPAN, DEFAULT_ROOT_CHORD, DEFAULT_DELTA_MAX,
        DEFAULT_RHO_SKIN, DEFAULT_BETA, DEFAULT_PAYLOAD,
        DEFAULT_W_INITIAL_SPAR,
    )
except ImportError:
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..'))
    from src.spar_design.load_calculator import LoadCalculator
    from src.spar_design.stiffness_optimizer import optimize_stiffness
    from src.spar_design.mandrel_dp import mandrel_selection_dp, DPResult
    from src.spar_design.mandrel import MANDREL_LIST, Mandrel
    from src.spar_design.layup_optimizer import LayupOptimizer
    from src.spar_design.config import (
        GRID_SIZE, MAX_ITER, CONV_TOL, RELAX_ALPHA,
        DEFAULT_SPAN, DEFAULT_ROOT_CHORD, DEFAULT_DELTA_MAX,
        DEFAULT_RHO_SKIN, DEFAULT_BETA, DEFAULT_PAYLOAD,
        DEFAULT_W_INITIAL_SPAR,
    )


@dataclass
class ConvergenceResult:
    """収束ループの最終出力。"""
    dp_result: DPResult                 # 最終マンドレル配置
    EI_opt: np.ndarray                  # 最適 EI 分布 [N·mm²]
    w_spar_dist: np.ndarray             # 最終桁重量分布 [kg/mm]
    total_spar_weight: float            # 片翼桁重量 [kg]
    n_iter: int                         # 実際の反復回数
    converged: bool                     # 収束フラグ
    W_total_kg: float                   # 最終全備重量 [kg]（片翼 × 2 + ペイロード）
    history: list[float]                # 各反復の桁重量 [kg]

    def print_summary(self) -> None:
        print(f"\n{'='*70}")
        print(f"  収束ループ結果")
        print(f"{'='*70}")
        print(f"  収束: {'✓ 収束' if self.converged else '✗ 未収束（最大反復到達）'}")
        print(f"  反復回数    : {self.n_iter}")
        print(f"  片翼桁重量  : {self.total_spar_weight*1000:.1f} g")
        print(f"  全備重量    : {self.W_total_kg:.2f} kg")
        print(f"  重量履歴    : {[f'{w*1000:.1f}g' for w in self.history]}")
        self.dp_result.print_summary()


def _build_spar_weight_distribution(
    dp_result: DPResult,
    y_mm: np.ndarray,
) -> np.ndarray:
    """
    DPResult の各セグメント重量からスパン方向の分布 [kg/mm] を構築する。

    各グリッド点にセグメントの weight_per_m を割り当て，
    重複区間の追加重量は均等に分散する。
    """
    w_dist = np.zeros(len(y_mm))

    for seg in dp_result.segments:
        # セグメントのカバー範囲のグリッド点インデックスを特定
        i_start = int(round(seg.start_y / GRID_SIZE))
        i_end   = int(round(seg.end_y   / GRID_SIZE))
        i_start = max(0, min(i_start, len(y_mm) - 1))
        i_end   = max(0, min(i_end,   len(y_mm) - 1))

        # セグメントの線密度 [kg/mm]
        w_per_mm = seg.layup.weight_per_m / 1000.0

        # 重複区間を含む範囲に均等配分
        n_pts = i_end - i_start + 1
        if n_pts <= 0:
            continue

        overlap_per_mm = seg.overlap_weight / max(seg.length, 1.0)

        for i in range(i_start, i_end + 1):
            w_dist[i] = max(w_dist[i], w_per_mm + overlap_per_mm)

    return w_dist


def run_convergence_loop(
    span_mm: float = DEFAULT_SPAN,
    root_chord_mm: float = DEFAULT_ROOT_CHORD,
    delta_max_mm: float = DEFAULT_DELTA_MAX,
    rho_skin_kg_m2: float = DEFAULT_RHO_SKIN,
    v_ms: float = 7.5,
    rho_air: float = 1.154,
    beta: float = DEFAULT_BETA,
    payload_kg: float = DEFAULT_PAYLOAD,
    w_spar_init_kg: float = DEFAULT_W_INITIAL_SPAR,
    relax: float = RELAX_ALPHA,
    max_iter: int = MAX_ITER,
    conv_tol: float = CONV_TOL,
    mandrel_list: list[Mandrel] | None = None,
    verbose: bool = True,
) -> ConvergenceResult:
    """
    重量収束ループを実行し，最終設計を返す。

    Args:
        span_mm         : 片翼スパン [mm]
        root_chord_mm   : 翼根コード長 [mm]
        delta_max_mm    : 翼端たわみ許容値 [mm]
        rho_skin_kg_m2  : 外皮面密度 [kg/m²]
        v_ms            : 飛行速度 [m/s]
        rho_air         : 空気密度 [kg/m³]
        beta            : TR-797 循環分布係数
        payload_kg      : ペイロード重量（パイロット＋機構）[kg]
        w_spar_init_kg  : 桁重量初期推定値（片翼）[kg]
        relax           : Picard 緩和係数 α
        max_iter        : 最大反復回数
        conv_tol        : 収束判定閾値 [kg]
        mandrel_list    : 利用可能なマンドレルリスト
        verbose         : True で各反復の状況を表示

    Returns:
        ConvergenceResult
    """
    if mandrel_list is None:
        mandrel_list = MANDREL_LIST

    opt  = LayupOptimizer()
    calc = LoadCalculator(
        span_mm        = span_mm,
        root_chord_mm  = root_chord_mm,
        rho_skin_kg_m2 = rho_skin_kg_m2,
        v_ms           = v_ms,
        rho_air        = rho_air,
    )
    y_mm = calc.y_mm

    # 初期桁重量分布（均一）
    w_spar_kg_per_mm = np.full(len(y_mm), w_spar_init_kg / span_mm)
    w_spar_total = w_spar_init_kg

    history: list[float] = [w_spar_total]
    converged = False
    n_iter    = 0
    dp_result = None
    EI_opt    = None

    if verbose:
        print(f"\n{'='*70}")
        print(f"  収束ループ開始  (span={span_mm}mm, δ_max={delta_max_mm}mm)")
        print(f"  ペイロード={payload_kg}kg, 緩和α={relax}, 収束閾値={conv_tol}kg")
        print(f"{'='*70}")

    for iteration in range(1, max_iter + 1):
        n_iter = iteration

        # ------ Step 1: 全備重量の推定 --------------------------------
        # 片翼桁重量 × 2 + ペイロード（二次構造は LoadCalculator 内で計算）
        W_total_kg = payload_kg + w_spar_total * 2.0

        # ------ Step 2: 荷重・曲げモーメント --------------------------
        M_Nmm, _ = calc.moment_distribution_Nmm(
            W_total_kg       = W_total_kg,
            beta             = beta,
            w_spar_kg_per_mm = w_spar_kg_per_mm,
        )

        # ------ Step 3: 最適 EI 分布 ---------------------------------
        EI_opt = optimize_stiffness(
            M_Nmm        = M_Nmm,
            y_mm         = y_mm,
            delta_max_mm = delta_max_mm,
            mandrel_list = mandrel_list,
            opt          = opt,
            verbose      = False,
        )

        # ------ Step 4: マンドレル DP --------------------------------
        dp_result = mandrel_selection_dp(
            EI_req_dist  = EI_opt,
            span_mm      = span_mm,
            mandrel_list = mandrel_list,
            opt          = opt,
        )

        if not dp_result.feasible:
            if verbose:
                print(f"  反復{iteration:3d}: DP 実行可能解なし → ループ終了")
            break

        w_spar_new_total = dp_result.total_spar_weight

        # ------ Step 5: Picard 緩和 -----------------------------------
        w_spar_relaxed = relax * w_spar_new_total + (1.0 - relax) * w_spar_total

        # 新しい桁重量分布を構築
        w_spar_new_dist = _build_spar_weight_distribution(dp_result, y_mm)
        # 総重量を緩和値にスケール（分布形状は DP 結果に従う）
        dist_total = float(np.trapz(w_spar_new_dist, y_mm))
        if dist_total > 0:
            w_spar_new_dist = w_spar_new_dist * (w_spar_relaxed / dist_total)

        delta_w = abs(w_spar_relaxed - w_spar_total)

        if verbose:
            print(f"  反復{iteration:3d}: W_total={W_total_kg:.2f}kg, "
                  f"W_spar={w_spar_relaxed*1000:.1f}g, Δw={delta_w*1000:.2f}g")

        # 更新
        w_spar_kg_per_mm = w_spar_new_dist
        w_spar_total     = w_spar_relaxed
        history.append(w_spar_total)

        # ------ Step 6: 収束判定 -------------------------------------
        if delta_w < conv_tol:
            converged = True
            if verbose:
                print(f"\n  ✓ 収束（{iteration}反復, Δw={delta_w*1000:.3f}g < {conv_tol*1000:.3f}g）")
            break

    if not converged and verbose:
        print(f"\n  ✗ 最大反復({max_iter})到達（未収束）")

    # 最終全備重量
    W_total_final = payload_kg + w_spar_total * 2.0

    return ConvergenceResult(
        dp_result          = dp_result,
        EI_opt             = EI_opt,
        w_spar_dist        = w_spar_kg_per_mm,
        total_spar_weight  = w_spar_total,
        n_iter             = n_iter,
        converged          = converged,
        W_total_kg         = W_total_final,
        history            = history,
    )


if __name__ == "__main__":
    result = run_convergence_loop(verbose=True)
    result.print_summary()
