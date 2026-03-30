"""
mandrel.py — マンドレルデータクラスと利用可能リスト
====================================================
仕様書 Section 2.2 に対応。
全寸法 mm 単位。
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Mandrel:
    """マンドレル1本を表すイミュータブルデータクラス。"""
    id: int            # 識別番号（0〜9）
    diameter: float    # 外径 [mm]（= 巻き付け内径）
    max_length: float  # 最大長 [mm]

    def __repr__(self) -> str:
        return f"Mandrel(φ{self.diameter:.0f}, max={self.max_length:.0f}mm)"


# -------------------------------------------------------
# 利用可能なマンドレル一覧（仕様書 Table 2.2）
# 翼根側（大径）→ 翼端側（小径）の順
# -------------------------------------------------------
MANDREL_LIST: list[Mandrel] = [
    Mandrel(id=0, diameter= 45, max_length=4000),
    Mandrel(id=1, diameter= 50, max_length=4000),
    Mandrel(id=2, diameter= 55, max_length=4000),
    Mandrel(id=3, diameter= 60, max_length=4000),
    Mandrel(id=4, diameter= 70, max_length=4000),
    Mandrel(id=5, diameter= 80, max_length=4000),
    Mandrel(id=6, diameter= 90, max_length=4000),
    Mandrel(id=7, diameter=100, max_length=2000),
    Mandrel(id=8, diameter=120, max_length=2000),
    Mandrel(id=9, diameter=130, max_length=2000),
]

# 直径→Mandrel の逆引き辞書
_DIAMETER_TO_MANDREL: dict[float, Mandrel] = {m.diameter: m for m in MANDREL_LIST}


def get_mandrel_by_diameter(diameter: float) -> Mandrel:
    """直径 [mm] からMandrelを取得。見つからなければ ValueError。"""
    if diameter not in _DIAMETER_TO_MANDREL:
        available = [m.diameter for m in MANDREL_LIST]
        raise ValueError(f"φ{diameter} mm のマンドレルは存在しません。利用可能: {available}")
    return _DIAMETER_TO_MANDREL[diameter]


if __name__ == "__main__":
    print("利用可能なマンドレル一覧:")
    for m in MANDREL_LIST:
        print(f"  ID{m.id}: {m}")
