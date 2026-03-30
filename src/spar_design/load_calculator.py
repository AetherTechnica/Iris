"""
load_calculator.py — 荷重・曲げモーメント計算（mm 単位）
=========================================================
AerodynamicsAnalyzer（TR-797）を mm 単位でラップし，
正味荷重分布と曲げモーメント分布を返す。

【荷重モデル】
  L_net(y) = L_aero(y) − g × [w_spar(y) + w_secondary(y)]

  L_aero    : TR-797 揚力分布 [N/mm]
  w_spar    : 桁の線密度 [kg/mm]（DP後に更新）
  w_secondary: 外皮＋リブの線密度 [kg/mm]
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

try:
    from .aerodynamics import AerodynamicsAnalyzer, AeroParams
    from .secondary_weight import secondary_weight_distribution
    from .deflection_calc import bending_moment
    from .config import GRID_SIZE
except ImportError:
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..'))
    from src.spar_design.aerodynamics import AerodynamicsAnalyzer, AeroParams
    from src.spar_design.secondary_weight import secondary_weight_distribution
    from src.spar_design.deflection_calc import bending_moment
    from src.spar_design.config import GRID_SIZE

_G = 9.80665  # m/s²


class LoadCalculator:
    """
    与えられた全備重量・機体パラメータから
    正味荷重分布と曲げモーメント分布を計算するクラス。
    """

    def __init__(
        self,
        span_mm: float,
        root_chord_mm: float,
        rho_skin_kg_m2: float,
        v_ms: float,
        rho_air: float,
        n_aero_segments: int = 100,
    ) -> None:
        """
        Args:
            span_mm         : 片翼スパン [mm]
            root_chord_mm   : 翼根コード長 [mm]
            rho_skin_kg_m2  : 外皮面密度 [kg/m²]
            v_ms            : 飛行速度 [m/s]
            rho_air         : 空気密度 [kg/m³]
            n_aero_segments : TR-797 の分割数
        """
        self.span_mm       = span_mm
        self.root_chord_mm = root_chord_mm
        self.rho_skin      = rho_skin_kg_m2
        self.v_ms          = v_ms
        self.rho_air       = rho_air
        self.n_aero        = n_aero_segments

        # 50mm 刻みの計算グリッド
        self.y_mm = np.arange(0.0, span_mm + GRID_SIZE, GRID_SIZE)

    # ----------------------------------------------------------
    # 揚力分布（TR-797）
    # ----------------------------------------------------------
    def lift_distribution_N_per_mm(
        self,
        W_total_kg: float,
        beta: float,
    ) -> np.ndarray:
        """
        TR-797 法で揚力分布を計算し，50mm グリッドに補間して返す。

        Args:
            W_total_kg: 全備重量 [kg]
            beta      : 循環分布係数

        Returns:
            L_aero [N/mm]（y_mm と同 shape）
        """
        params = AeroParams(
            lift_N      = W_total_kg * _G,
            span_m      = self.span_mm / 1000.0,
            v_ms        = self.v_ms,
            rho         = self.rho_air,
            n_segments  = self.n_aero,
        )
        solver = AerodynamicsAnalyzer(params)
        res    = solver.solve(beta)

        if not res:
            raise RuntimeError("TR-797 ソルバーが収束しませんでした")

        # m 単位 [N/m] → mm 単位 [N/mm]，グリッドに線形補間
        y_aero_mm = res["span_y_m"] * 1000.0
        L_aero_m  = res["lift_dist_N_per_m"]

        f = interp1d(y_aero_mm, L_aero_m / 1000.0,
                     kind="linear", fill_value="extrapolate")
        return f(self.y_mm)

    # ----------------------------------------------------------
    # 正味荷重
    # ----------------------------------------------------------
    def net_load_N_per_mm(
        self,
        W_total_kg: float,
        beta: float,
        w_spar_kg_per_mm: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        空力荷重から自重を差し引いた正味荷重分布 [N/mm]。

        Args:
            W_total_kg      : 全備重量 [kg]
            beta            : 循環分布係数
            w_spar_kg_per_mm: 桁の分布重量 [kg/mm]（None=無視）

        Returns:
            L_net [N/mm]
        """
        L_aero = self.lift_distribution_N_per_mm(W_total_kg, beta)

        # 二次構造重量 [kg/mm] → [N/mm]
        w_sec = secondary_weight_distribution(
            self.y_mm, self.span_mm, self.root_chord_mm, self.rho_skin
        )
        L_sec = w_sec * _G  # [N/mm]

        # 桁重量 [N/mm]
        if w_spar_kg_per_mm is not None:
            L_spar = w_spar_kg_per_mm * _G
        else:
            L_spar = np.zeros_like(self.y_mm)

        return L_aero - L_sec - L_spar

    # ----------------------------------------------------------
    # 曲げモーメント
    # ----------------------------------------------------------
    def moment_distribution_Nmm(
        self,
        W_total_kg: float,
        beta: float,
        w_spar_kg_per_mm: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        正味荷重から曲げモーメント分布を計算する。

        Returns:
            (M_dist [N·mm], L_net [N/mm])
        """
        L_net = self.net_load_N_per_mm(W_total_kg, beta, w_spar_kg_per_mm)
        _, M  = bending_moment(L_net, self.y_mm)
        return M, L_net


if __name__ == "__main__":
    calc = LoadCalculator(
        span_mm        = 10000.0,
        root_chord_mm  = 300.0,
        rho_skin_kg_m2 = 0.05,
        v_ms           = 7.5,
        rho_air        = 1.154,
    )

    M, L_net = calc.moment_distribution_Nmm(W_total_kg=82.0, beta=0.9)

    print(f"翼根曲げモーメント: {M[0]:.3e} N·mm")
    print(f"最大正味荷重      : {L_net.max():.4f} N/mm")
    print(f"翼端正味荷重      : {L_net[-1]:.4f} N/mm")
