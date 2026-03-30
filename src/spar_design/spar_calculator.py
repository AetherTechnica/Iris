"""
spar_calculator.py — CFRP円管断面の剛性・重量計算
===================================================
Hemere/src/core/spar_calculator.py を Iris 向けに移植。
Hemere 側は一切変更しない。

【11層積層テンプレート】
  Index | 種別      | 角度 [deg] | 材料
  ------+-----------+------------+--------
    0   | Protect内 |   90       | 24t
    1   | Torque    |   90       | 40t (45°繊維)
    2   | Base      |   90       | 40t (0°繊維)
    3   | Cap1      |   50       | 40t (0°繊維)
    4   | Cap2      |   45       | 40t (0°繊維)
    5   | Cap3      |   40       | 40t (0°繊維)
    6   | Cap4      |   35       | 40t (0°繊維)
    7   | Cap5      |   30       | 40t (0°繊維)
    8   | Cap6      |   25       | 40t (0°繊維)
    9   | Cap7      |   20       | 40t (0°繊維)
   10   | Protect外 |   90       | 24t

【出力単位】
  EI     : kgf·mm²（LayupOptimizerで N·mm² に変換）
  Weight : kg/m
  Thickness: mm
"""

from __future__ import annotations
import numpy as np


class SparCalculator:
    """CFRPパイプの断面剛性・重量を積層構成から計算するクラス。"""

    # 材料ヤング率 [kgf/mm²]
    # 0:24t_0°  1:24t_45°  2:24t_90°  3:40t_0°  4:40t_45°  5:40t_90°
    _MAT_E_KGF = np.array([13000, 1900, 900, 22000, 1900, 800], dtype=float)

    # 各層の有効角度幅 [deg]（部分積層の円弧カバー率）
    _PLY_ANGLES = np.array([90, 90, 90, 50, 45, 40, 35, 30, 25, 20, 90], dtype=float)

    # 1ply あたりの厚み [mm]
    _T_24T = 0.125
    _T_40T = 0.111

    # 線密度係数 [kg/mm³]（断面積1mm²・長さ1mmあたりの質量）
    _RHO_24T = 0.001496
    _RHO_40T = 0.001559

    N_LAYERS = 11

    def calculate_spec(
        self,
        ply_counts: np.ndarray,
        diameter_mm: float,
    ) -> tuple[float, float, float]:
        """
        積層数配列と直径から断面性能を計算する。

        Args:
            ply_counts : shape(11,) の整数配列。各層の積層枚数。
            diameter_mm: マンドレル直径 [mm]（= 最内層の内径）

        Returns:
            (EI [kgf·mm²], weight [kg/m], total_thickness [mm])
        """
        if len(ply_counts) != self.N_LAYERS:
            raise ValueError(
                f"ply_countsの要素数は{self.N_LAYERS}である必要があります "
                f"（受け取った: {len(ply_counts)}）"
            )

        # 1. 各層の厚み [mm]
        thickness = ply_counts * self._T_40T
        thickness[0]  = ply_counts[0]  * self._T_24T   # Protect 内（24t）
        thickness[10] = ply_counts[10] * self._T_24T   # Protect 外（24t）

        # 2. 各層の内径・外径を内側から積算
        inner_dia = np.zeros(self.N_LAYERS)
        outer_dia = np.zeros(self.N_LAYERS)
        r = diameter_mm
        for i in range(self.N_LAYERS):
            inner_dia[i] = r
            outer_dia[i] = r + 2 * thickness[i]
            r = outer_dia[i]

        # 3. 断面二次モーメント（部分積層を考慮）
        #    Ix = (D_outer⁴ - D_inner⁴) × angle_factor / 16
        angle_rad = self._PLY_ANGLES * np.pi / 180.0
        ix_factors = (angle_rad / 2.0 + np.sin(2 * angle_rad) / 4.0) / 16.0
        Ix = (outer_dia**4 - inner_dia**4) * ix_factors

        # 4. 各層のヤング率を割り当て
        E_vec = np.zeros(self.N_LAYERS)
        for i in range(self.N_LAYERS):
            layer = i + 1  # 1-based
            if layer in (1, 11):
                E_vec[i] = self._MAT_E_KGF[2]   # 24t_90°
            elif layer == 2:
                E_vec[i] = self._MAT_E_KGF[4]   # 40t_45°
            else:
                E_vec[i] = self._MAT_E_KGF[3]   # 40t_0°

        # 5. 曲げ剛性 EI [kgf·mm²]
        EI_kgf = float(np.sum(Ix * E_vec))

        # 6. 線密度 [kg/m]
        #    断面積の近似 = π/4 × (D_out² - D_in²) × (angle/90°)
        area_factor = (np.pi / 4.0) * (outer_dia**2 - inner_dia**2) * (self._PLY_ANGLES / 90.0)
        rho_vec = np.where(
            np.isin(np.arange(self.N_LAYERS), [0, 10]),
            self._RHO_24T,
            self._RHO_40T,
        )
        weight_kg_m = float(np.sum(rho_vec * area_factor)) * 1000.0  # mm→m 換算

        total_thickness = float(np.sum(thickness))

        return EI_kgf, weight_kg_m, total_thickness


if __name__ == "__main__":
    calc = SparCalculator()
    test_ply = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    test_dia = 100.0
    ei, w, t = calc.calculate_spec(test_ply, test_dia)
    print(f"EI      = {ei:.4e} kgf·mm²  ({ei * 9.80665:.4e} N·mm²)")
    print(f"Weight  = {w:.4f} kg/m")
    print(f"Thick   = {t:.3f} mm")
