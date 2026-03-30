"""
aerodynamics.py — TR-797法による揚力分布計算
=============================================
Hemere/src/aerodynamics/aerodynamics_analyzer.py を Iris 向けに移植。
Hemere 側は一切変更しない。

入力は m 単位（AerodynamicsAnalyzer の内部仕様に合わせる）。
外部インターフェース（LoadCalculator）が mm ↔ m 変換を担う。
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class AeroParams:
    """TR-797 ソルバーに渡す機体・環境パラメータ（m 単位）。"""
    lift_N: float        # 必要揚力 [N]  = W_total [kg] × g
    span_m: float        # 翼幅（片翼）[m]
    v_ms: float          # 飛行速度 [m/s]
    rho: float           # 空気密度 [kg/m³]
    n_segments: int = 100  # スパン方向分割数（多いほど精度↑）


class AerodynamicsAnalyzer:
    """
    TR-797 法に基づく揚力循環分布ソルバー。
    半翼を n_segments 等分し，循環分布を連立方程式で解く。
    """

    def __init__(self, params: AeroParams) -> None:
        self.p = params
        le = params.span_m / 2.0          # 半スパン [m]
        N = params.n_segments
        dS = np.ones(N) * le / N / 2.0
        self.le  = le
        self.dS  = dS
        self.y   = np.linspace(dS[0], le - dS[-1], N)  # 制御点位置 [m]
        self._precompute()

    def _precompute(self) -> None:
        y, dS, le = self.y, self.dS, self.le
        phi = np.zeros_like(y)
        z   = np.zeros_like(y)

        yc = y[:, None];  zc = z[:, None];  pc = phi[:, None]

        yd  = (yc - y) * np.cos(phi) + (zc - z) * np.sin(phi)
        zd  = -(yc - y) * np.sin(phi) + (zc - z) * np.cos(phi)
        y2d = (yc + y) * np.cos(phi) + (zc - z) * np.sin(phi)
        z2d = -(yc + y) * np.sin(phi) + (zc - z) * np.cos(phi)

        Rp  = (yd - dS)**2 + zd**2
        Rm  = (yd + dS)**2 + zd**2
        R2p = (y2d + dS)**2 + z2d**2
        R2m = (y2d - dS)**2 + z2d**2

        t1 = (-(yd - dS)/Rp + (yd + dS)/Rm) * np.cos(pc - phi)
        t2 = (-zd/Rp + zd/Rm) * np.sin(pc - phi)
        t3 = (-(y2d - dS)/R2m + (y2d + dS)/R2p) * np.cos(pc + phi)
        t4 = (-z2d/R2m + z2d/R2p) * np.sin(pc + phi)

        Q  = (t1 + t2 + t3 + t4) / (2 * np.pi)
        ds = dS / le
        self.A  = np.pi * Q * le * ds[:, None]
        self.ds = ds
        self.eta = self.y / le

    def solve(self, beta: float) -> dict:
        """
        循環分布を解く。

        Args:
            beta: 揚力モーメント低減率（楕円分布=1.0）

        Returns:
            {lift_dist_N_per_m, span_y_m, gamma}
        """
        N  = self.p.n_segments
        A  = self.A
        cv = 2.0 * self.ds
        bv = 1.5 * np.pi * self.ds * self.eta

        S = np.zeros((N + 2, N + 2))
        S[:N, :N] = A + A.T
        S[:N, N]  = -cv;  S[:N, N+1] = -bv
        S[N,  :N] = -cv;  S[N+1, :N] = -bv

        rhs = np.zeros(N + 2)
        rhs[N] = -1.0;  rhs[N+1] = -beta

        try:
            sol = np.linalg.solve(S, rhs)
        except np.linalg.LinAlgError:
            return {}

        g = sol[:N]
        p = self.p
        Gamma = (p.lift_N / (2 * self.le * p.rho * p.v_ms)) * g
        lift_dist = p.rho * p.v_ms * Gamma          # [N/m]

        return {
            "lift_dist_N_per_m": lift_dist,
            "span_y_m": self.y,
            "gamma": g,
        }
