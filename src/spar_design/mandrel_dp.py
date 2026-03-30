"""
mandrel_dp.py — 動的計画法によるマンドレル最適配置
====================================================
仕様書 Section 5 に対応。

【状態空間】
  dp[y_idx][m_id] = 位置 y まで到達し，
                    最後に使用したマンドレルが m である場合の最小総重量 [kg]

【遷移】
  (y_current, m_current) → (y_end, m_next)
  ・セグメント長: OVERLAP_LENGTH+GRID_SIZE 〜 max_length, GRID_SIZE刻み
  ・y_advance = length - OVERLAP_LENGTH（重複分を差し引いた実質進行距離）
  ・嵌合条件: OD(m_next, EI_max) + CLEARANCE ≤ D(m_current)

【重複区間の重量（Section 3.2）】
  W_overlap_outer = outer_layup.weight_per_m × 0.250      [kg]
  W_balsa         = ρ_balsa × A_gap × BALSA_FILL_LENGTH   [kg]
  A_gap = π/4 × (D_outer² - OD_inner²)

【注意】
  翼根側セグメント = outer（ソケット），翼端側セグメント = inner（挿入側）
  DP は翼根 (y=0) から翼端 (y=span) 方向へ進む。
  翼端側マンドレルの径 ≤ 翼根側マンドレルの径（嵌合条件より）。
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    from .mandrel import Mandrel, MANDREL_LIST
    from .layup_optimizer import LayupOptimizer, LayupResult
    from .config import (
        GRID_SIZE, OVERLAP_LENGTH, CLEARANCE,
        RHO_BALSA, BALSA_FILL_LENGTH,
    )
except ImportError:
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..'))
    from src.spar_design.mandrel import Mandrel, MANDREL_LIST
    from src.spar_design.layup_optimizer import LayupOptimizer, LayupResult
    from src.spar_design.config import (
        GRID_SIZE, OVERLAP_LENGTH, CLEARANCE,
        RHO_BALSA, BALSA_FILL_LENGTH,
    )

INF = float('inf')


# ============================================================
# データ構造
# ============================================================

@dataclass
class Segment:
    """スパー1セグメント（1本の桁）を表す。"""
    start_y: float         # mm（翼根側端部の実質位置）
    end_y: float           # mm（翼端側端部）
    mandrel: Mandrel
    layup: LayupResult
    length: float          # mm（物理的な管長, 重複 250mm 含む）
    weight: float          # kg（このセグメント単体の桁重量）
    overlap_weight: float  # kg（かんざし重複区間＋バルサ補填の追加重量）

    @property
    def total_weight(self) -> float:
        return self.weight + self.overlap_weight

    def __repr__(self) -> str:
        return (
            f"Segment(y={self.start_y:.0f}〜{self.end_y:.0f}mm, "
            f"φ{self.mandrel.diameter:.0f}, "
            f"EI={self.layup.EI_Nmm2:.2e} N·mm², "
            f"W={self.total_weight*1000:.1f}g)"
        )


@dataclass
class DPResult:
    """DP最適化の出力をまとめるデータクラス。"""
    segments: list[Segment]
    total_spar_weight: float        # kg（桁＋重複区間の合計）
    feasible: bool

    def print_summary(self) -> None:
        print(f"\n{'='*70}")
        print(f"  スパー最適配置結果  合計桁重量: {self.total_spar_weight*1000:.1f} g")
        print(f"{'='*70}")
        print(f"{'#':>2}  {'y範囲 [mm]':>14}  {'φ[mm]':>6}  "
              f"{'管長[mm]':>8}  {'EI[N·mm²]':>12}  "
              f"{'積層':>20}  {'重量[g]':>8}  {'重複[g]':>8}")
        print("-" * 90)
        for i, seg in enumerate(self.segments):
            print(
                f"{i:>2}  {seg.start_y:>6.0f}〜{seg.end_y:>6.0f}  "
                f"{seg.mandrel.diameter:>6.0f}  "
                f"{seg.length:>8.0f}  "
                f"{seg.layup.EI_Nmm2:>12.3e}  "
                f"{str(seg.layup.ply_counts.tolist()):>20}  "
                f"{seg.weight*1000:>8.1f}  "
                f"{seg.overlap_weight*1000:>8.1f}"
            )
        print("=" * 70)


# ============================================================
# 嵌合判定・重複重量計算
# ============================================================

def can_fit_inside(
    inner: Mandrel,
    outer: Mandrel,
    inner_layup: LayupResult,
) -> bool:
    """
    inner桁が outer桁のソケットに嵌合可能か判定。

    Args:
        inner      : 翼端側（挿入される側）マンドレル
        outer      : 翼根側（ソケット側）マンドレル
        inner_layup: inner桁の積層結果（OD を参照）

    Returns:
        True = 嵌合可能
    """
    OD_inner = inner_layup.OD  # inner桁の外径 [mm]
    ID_outer = outer.diameter  # outer桁の内径 [mm]（= マンドレル径）
    return OD_inner + CLEARANCE <= ID_outer


def compute_overlap_weight(
    outer_mandrel: Mandrel,
    outer_layup: LayupResult,
    inner_layup: LayupResult,
) -> tuple[float, float]:
    """
    かんざし重複区間の追加重量を計算する（仕様書 Section 3.2）。

    Args:
        outer_mandrel: 翼根側（ソケット）マンドレル
        outer_layup  : 翼根側セグメントの積層結果
        inner_layup  : 翼端側セグメントの積層結果

    Returns:
        (W_overlap_outer [kg], W_balsa [kg])
    """
    # 外側桁の 250mm 分の追加重量
    W_overlap_outer = outer_layup.weight_per_m * OVERLAP_LENGTH / 1000.0  # kg

    # 径差補填バルサ重量
    ID_outer = outer_mandrel.diameter   # 外側桁の内径（= 外側マンドレル径） [mm]
    OD_inner = inner_layup.OD           # 内側桁の外径 [mm]
    A_gap = np.pi / 4.0 * (ID_outer**2 - OD_inner**2)  # mm²
    # A_gap が負になる（OD_inner > ID_outer）場合は嵌合不可 → バルサなし
    W_balsa = RHO_BALSA * max(A_gap, 0.0) * BALSA_FILL_LENGTH  # kg

    return W_overlap_outer, W_balsa


# ============================================================
# DP 本体
# ============================================================

def mandrel_selection_dp(
    EI_req_dist: np.ndarray,
    span_mm: float,
    mandrel_list: list[Mandrel] | None = None,
    opt: LayupOptimizer | None = None,
) -> DPResult:
    """
    動的計画法でマンドレル配置を最適化する。

    Args:
        EI_req_dist : shape (n_stations,) の配列。
                      EI_req_dist[i] = y = i*GRID_SIZE [mm] での必要 EI [N·mm²]。
        span_mm     : 片翼スパン [mm]。
        mandrel_list: 利用可能なマンドレルリスト（None で MANDREL_LIST 使用）。
        opt         : LayupOptimizer インスタンス（None で新規作成）。

    Returns:
        DPResult
    """
    if mandrel_list is None:
        mandrel_list = MANDREL_LIST
    if opt is None:
        opt = LayupOptimizer()

    n_stations = int(span_mm // GRID_SIZE) + 1
    n_mandrels = len(mandrel_list)

    # DP テーブルと親情報テーブル
    dp     = np.full((n_stations, n_mandrels), INF)
    parent = [[None] * n_mandrels for _ in range(n_stations)]

    # 初期化: 翼根（y=0）は全マンドレルでコスト 0
    dp[0, :] = 0.0

    # ----------------------------------------------------------
    # 遷移ループ
    # ----------------------------------------------------------
    for y_idx in range(n_stations - 1):
        y_current = y_idx * GRID_SIZE  # mm

        for m_cur_id, m_cur in enumerate(mandrel_list):
            if dp[y_idx, m_cur_id] == INF:
                continue  # この状態には未到達

            cost_so_far = dp[y_idx, m_cur_id]

            # 外側桁（m_cur）の積層情報を取得（重複重量計算に使用）
            # y_current での EI_req を基準に（翼根側なので高め）
            EI_at_cur = EI_req_dist[y_idx] if EI_req_dist[y_idx] > 0 else 1.0
            outer_layup = opt.optimize(EI_at_cur, m_cur.diameter)

            # --- 次セグメント（m_nxt）を探索 ---
            for m_nxt_id, m_nxt in enumerate(mandrel_list):

                # セグメント長を GRID_SIZE 刻みで探索
                len_min = OVERLAP_LENGTH + GRID_SIZE  # 300 mm
                len_max = int(m_nxt.max_length)

                # range の上限は len_max + 1（+GRID_SIZE だと1ステップ超過する）
                for length in range(len_min, len_max + 1, GRID_SIZE):
                    y_advance = length - OVERLAP_LENGTH
                    y_end     = y_current + y_advance

                    # スパン超過時はクランプ
                    if y_end > span_mm:
                        y_end  = span_mm
                        length = int(y_end - y_current) + OVERLAP_LENGTH

                    y_end_idx = int(round(y_end / GRID_SIZE))

                    # この区間の最大 EI
                    i_start = y_idx
                    i_end   = min(y_end_idx, n_stations - 1)
                    max_EI  = float(np.max(EI_req_dist[i_start:i_end + 1]))
                    if max_EI <= 0:
                        max_EI = 1.0

                    # 積層設計（max_EI を満たす最軽量構成）
                    layup_nxt = opt.optimize(max_EI, m_nxt.diameter)
                    if not layup_nxt.feasible:
                        continue  # この直径では EI 不足

                    # 嵌合条件チェック（初回セグメント以外）
                    if y_idx > 0 and not can_fit_inside(m_nxt, m_cur, layup_nxt):
                        continue

                    # セグメント重量
                    W_seg = layup_nxt.weight_per_m * length / 1000.0  # kg

                    # 重複区間の追加重量（初回セグメントは 0）
                    if y_idx == 0:
                        W_ov, W_bl = 0.0, 0.0
                    else:
                        W_ov, W_bl = compute_overlap_weight(
                            m_cur, outer_layup, layup_nxt
                        )
                    W_overlap = W_ov + W_bl

                    # コスト更新
                    new_cost = cost_so_far + W_seg + W_overlap

                    if new_cost < dp[y_end_idx, m_nxt_id]:
                        dp[y_end_idx, m_nxt_id] = new_cost
                        parent[y_end_idx][m_nxt_id] = {
                            'prev_y_idx'    : y_idx,
                            'prev_m_id'     : m_cur_id,
                            'mandrel'       : m_nxt,
                            'layup'         : layup_nxt,
                            'length'        : float(length),
                            'weight'        : W_seg,
                            'overlap_weight': W_overlap,
                        }

                    # スパン端に到達したらそれ以上の長さは不要
                    if y_end >= span_mm:
                        break

    # ----------------------------------------------------------
    # 最適解の復元（バックトラック）
    # ----------------------------------------------------------
    span_idx = n_stations - 1

    # 翼端での最小コストのマンドレルを選択
    min_cost_at_tip = np.min(dp[span_idx, :])
    if min_cost_at_tip == INF:
        return DPResult(segments=[], total_spar_weight=INF, feasible=False)

    best_m_id = int(np.argmin(dp[span_idx, :]))

    segments: list[Segment] = []
    y_idx = span_idx
    m_id  = best_m_id

    while parent[y_idx][m_id] is not None:
        info = parent[y_idx][m_id]

        seg = Segment(
            start_y        = info['prev_y_idx'] * GRID_SIZE,
            end_y          = y_idx * GRID_SIZE,
            mandrel        = info['mandrel'],
            layup          = info['layup'],
            length         = info['length'],
            weight         = info['weight'],
            overlap_weight = info['overlap_weight'],
        )
        segments.append(seg)

        # 一つ前の状態へ
        next_y_idx = info['prev_y_idx']
        next_m_id  = info['prev_m_id']
        y_idx = next_y_idx
        m_id  = next_m_id

    segments.reverse()  # 翼根→翼端の順に並べ替え

    return DPResult(
        segments           = segments,
        total_spar_weight  = min_cost_at_tip,
        feasible           = True,
    )


# ============================================================
# 動作確認（小規模テスト）
# ============================================================
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  MandrelDP 動作テスト（span=2000mm, 単純EI分布）")
    print("=" * 70)

    # 簡易 EI 分布（翼根が高く，翼端に向かって線形減少）
    span     = 2000   # mm
    n_sta    = span // GRID_SIZE + 1
    y_arr    = np.arange(n_sta) * GRID_SIZE
    EI_root  = 5.0e10  # N·mm²（翼根）
    EI_tip   = 1.0e8   # N·mm²（翼端）
    EI_dist  = EI_root * (1 - y_arr / span) + EI_tip * (y_arr / span)

    print(f"スパン     : {span} mm")
    print(f"EI (翼根)  : {EI_root:.2e} N·mm²")
    print(f"EI (翼端)  : {EI_tip:.2e} N·mm²")
    print(f"計算ステーション数: {n_sta}")
    print()

    result = mandrel_selection_dp(EI_dist, span_mm=span)

    if result.feasible:
        result.print_summary()
    else:
        print("✗ 実行可能解なし（マンドレル径・EI範囲を見直してください）")
        sys.exit(1)
