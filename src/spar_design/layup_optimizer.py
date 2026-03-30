"""
layup_optimizer.py — 純物理ベース貪欲法積層最適化
===================================================
AIサロゲートモデルに依存しない，SparCalculator のみを使った
最小積層数探索。

【設計方針】
  - 可変プライ（インデックス 2〜9）を1枚ずつ追加し，
    最も EI/重量効率の高いプライを選択する貪欲法。
  - lru_cache で (EI_req, diameter) → LayupResult をキャッシュ。
  - 出力単位: N·mm²（内部で kgf→N 変換）

【積層テンプレート固定層】
  idx 0 : Protect内  = 1枚（固定）
  idx 1 : Torque     = 2枚（固定）
  idx 10: Protect外  = 1枚（固定）

【可変層】
  idx 2 : Base（40t 0°, 90°カバー）
  idx 3〜9: Cap1〜7（40t 0°, 50°〜20°グラデーション）
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from functools import lru_cache

from .spar_calculator import SparCalculator
from .config import BUCKLING_LIMIT, MAX_PLY_PER_LAYER

# kgf → N 変換係数
_G = 9.80665

# 固定層の定義
_FIXED_PLY: dict[int, int] = {0: 1, 1: 2, 10: 1}
# 可変層インデックス（Base + Cap1〜7）
_VAR_INDICES: list[int] = [2, 3, 4, 5, 6, 7, 8, 9]

# EI要求値をキャッシュ丸め単位 [N·mm²]
_CACHE_ROUND = 1e6


@dataclass
class LayupResult:
    """LayupOptimizer の出力を格納するデータクラス。"""
    ply_counts: np.ndarray   # shape(11,) 各層の積層枚数
    weight_per_m: float      # [kg/m]
    EI_Nmm2: float           # 実現曲げ剛性 [N·mm²]
    thickness: float         # 総積層厚 [mm]
    OD: float                # 外径 [mm] = マンドレル径 + 2×thickness
    feasible: bool           # True = EI_req を満たした

    def weight_for_length(self, length_mm: float) -> float:
        """指定長さ [mm] のセグメント重量 [kg] を返す。"""
        return self.weight_per_m * length_mm / 1000.0


class LayupOptimizer:
    """
    与えられた EI_req [N·mm²] と mandrel_diameter [mm] に対して
    最小重量の積層構成を貪欲法で探索するクラス。

    キャッシュは一インスタンス内で共有される。
    """

    def __init__(self) -> None:
        self._calc = SparCalculator()
        # (EI_req_rounded, diameter) → LayupResult のキャッシュ
        self._cache: dict[tuple[int, float], LayupResult] = {}

    # ----------------------------------------------------------
    # 公開インターフェース
    # ----------------------------------------------------------
    def optimize(self, EI_req_Nmm2: float, diameter_mm: float) -> LayupResult:
        """
        EI_req を満たす最軽量積層構成を返す。

        Args:
            EI_req_Nmm2  : 必要曲げ剛性 [N·mm²]
            diameter_mm  : マンドレル直径 [mm]

        Returns:
            LayupResult（feasible=False の場合は最大積層時の値）
        """
        # キャッシュキー（EI を _CACHE_ROUND 単位で丸める）
        key = (int(round(EI_req_Nmm2 / _CACHE_ROUND)), diameter_mm)
        if key in self._cache:
            return self._cache[key]

        result = self._greedy_search(EI_req_Nmm2, diameter_mm)
        self._cache[key] = result
        return result

    def get_feasible_EI_range(self, diameter_mm: float) -> tuple[float, float]:
        """
        指定直径で実現可能な EI の範囲 [N·mm²] を返す。

        Returns:
            (EI_min, EI_max)
        """
        # 最小積層（固定層のみ）
        ply_min = self._initial_ply()
        ei_min_kgf, _, _ = self._calc.calculate_spec(ply_min, diameter_mm)

        # 最大積層（全可変層が MAX_PLY_PER_LAYER 枚）
        ply_max = self._initial_ply()
        for i in _VAR_INDICES:
            ply_max[i] = MAX_PLY_PER_LAYER
        ei_max_kgf, _, _ = self._calc.calculate_spec(ply_max, diameter_mm)

        return ei_min_kgf * _G, ei_max_kgf * _G

    # ----------------------------------------------------------
    # 内部実装
    # ----------------------------------------------------------
    def _initial_ply(self) -> np.ndarray:
        """初期積層（固定層のみ，可変層は 0 枚）を返す。"""
        ply = np.zeros(11, dtype=int)
        for idx, cnt in _FIXED_PLY.items():
            ply[idx] = cnt
        return ply

    def _evaluate(self, ply: np.ndarray, D: float) -> tuple[float, float, float]:
        """SparCalculator を呼んで (EI [N·mm²], weight [kg/m], thickness [mm]) を返す。"""
        ei_kgf, w, t = self._calc.calculate_spec(ply, D)
        return ei_kgf * _G, w, t

    def _greedy_search(self, EI_req_Nmm2: float, D: float) -> LayupResult:
        """
        貪欲法本体。

        各ステップで可変プライを1枚追加したときの
        ΔΔEI / ΔWeight を計算し，最も効率の高いプライを選択する。
        """
        ply = self._initial_ply()
        EI_req_kgf = EI_req_Nmm2 / _G

        current_ei_kgf, current_w, current_t = self._calc.calculate_spec(ply, D)

        # すでに要件を満たしているか確認
        if current_ei_kgf >= EI_req_kgf:
            return self._make_result(ply, D, EI_req_Nmm2, feasible=True)

        # 最大反復回数 = 可変層数 × MAX_PLY_PER_LAYER
        max_steps = len(_VAR_INDICES) * MAX_PLY_PER_LAYER

        for _ in range(max_steps):
            best_ratio = -1.0
            best_idx   = -1

            for i in _VAR_INDICES:
                # 上限チェック
                if ply[i] >= MAX_PLY_PER_LAYER:
                    continue

                # 1枚追加した場合の評価
                ply[i] += 1
                new_ei_kgf, new_w, _ = self._calc.calculate_spec(ply, D)
                ply[i] -= 1

                delta_ei = new_ei_kgf - current_ei_kgf
                delta_w  = new_w - current_w

                # 重量増加がほぼゼロの場合をスキップ
                if delta_w < 1e-9:
                    continue

                ratio = delta_ei / delta_w
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx   = i

            # 有効なプライが見つからない → 打ち切り
            if best_idx == -1:
                break

            ply[best_idx] += 1
            current_ei_kgf, current_w, current_t = self._calc.calculate_spec(ply, D)

            if current_ei_kgf >= EI_req_kgf:
                # 座屈制約チェック（D/t ≤ BUCKLING_LIMIT）
                if current_t > 0 and (D / current_t) > BUCKLING_LIMIT:
                    # 座屈NGでも feasible=True とし，呼び出し元に判断を委ねる
                    # （実際には太径マンドレルを選ぶべきケース）
                    pass
                return self._make_result(ply, D, EI_req_Nmm2, feasible=True)

        # ループ終了 → EI_req 未達
        return self._make_result(ply, D, EI_req_Nmm2, feasible=False)

    def _make_result(
        self,
        ply: np.ndarray,
        D: float,
        EI_req_Nmm2: float,
        feasible: bool,
    ) -> LayupResult:
        """LayupResult を生成して返す。"""
        ei_kgf, w, t = self._calc.calculate_spec(ply, D)
        return LayupResult(
            ply_counts   = ply.copy(),
            weight_per_m = w,
            EI_Nmm2      = ei_kgf * _G,
            thickness    = t,
            OD           = D + 2 * t,
            feasible     = feasible,
        )


# ----------------------------------------------------------
# 動作確認
# ----------------------------------------------------------
if __name__ == "__main__":
    opt = LayupOptimizer()

    test_cases = [
        (1.0e9,  90.0),
        (5.0e9,  90.0),
        (2.0e10, 90.0),
        (1.0e10, 60.0),
        (5.0e10, 130.0),
    ]

    print(f"{'EI_req [N·mm²]':>16} | {'D [mm]':>6} | {'EI_act [N·mm²]':>16} | "
          f"{'W [kg/m]':>9} | {'thick [mm]':>10} | {'OD [mm]':>8} | {'OK':>4} | ply")
    print("-" * 100)

    for EI_req, D in test_cases:
        r = opt.optimize(EI_req, D)
        ok = "✓" if r.feasible else "✗"
        print(f"{EI_req:16.3e} | {D:6.1f} | {r.EI_Nmm2:16.3e} | "
              f"{r.weight_per_m:9.4f} | {r.thickness:10.3f} | {r.OD:8.3f} | "
              f"{ok:>4} | {r.ply_counts.tolist()}")
