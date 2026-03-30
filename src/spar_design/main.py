"""
main.py — Iris スパー最適設計メインスクリプト
================================================
コマンドライン引数で機体パラメータを指定し，
最適スパー配置・重量を出力する。

【実行例】
  python -m src.spar_design.main
  python -m src.spar_design.main --span 12000 --delta-max 2500 --payload 75
  python main.py --verbose
"""

from __future__ import annotations

import argparse
import sys
import numpy as np

try:
    from .convergence_loop import run_convergence_loop
    from .config import (
        DEFAULT_SPAN, DEFAULT_ROOT_CHORD, DEFAULT_DELTA_MAX,
        DEFAULT_RHO_SKIN, DEFAULT_BETA, DEFAULT_PAYLOAD,
        DEFAULT_W_INITIAL_SPAR, RELAX_ALPHA, MAX_ITER, CONV_TOL,
    )
except ImportError:
    import os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..'))
    from src.spar_design.convergence_loop import run_convergence_loop
    from src.spar_design.config import (
        DEFAULT_SPAN, DEFAULT_ROOT_CHORD, DEFAULT_DELTA_MAX,
        DEFAULT_RHO_SKIN, DEFAULT_BETA, DEFAULT_PAYLOAD,
        DEFAULT_W_INITIAL_SPAR, RELAX_ALPHA, MAX_ITER, CONV_TOL,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Iris — 人力飛行機主翼スパー最適設計ツール",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--span",         type=float, default=DEFAULT_SPAN,
                   help="片翼スパン [mm]")
    p.add_argument("--chord",        type=float, default=DEFAULT_ROOT_CHORD,
                   help="翼根コード長 [mm]")
    p.add_argument("--delta-max",    type=float, default=DEFAULT_DELTA_MAX,
                   help="翼端たわみ許容値 [mm]")
    p.add_argument("--rho-skin",     type=float, default=DEFAULT_RHO_SKIN,
                   help="外皮面密度 [kg/m²]")
    p.add_argument("--v",            type=float, default=7.5,
                   help="飛行速度 [m/s]")
    p.add_argument("--rho-air",      type=float, default=1.154,
                   help="空気密度 [kg/m³]")
    p.add_argument("--beta",         type=float, default=DEFAULT_BETA,
                   help="TR-797 循環分布係数")
    p.add_argument("--payload",      type=float, default=DEFAULT_PAYLOAD,
                   help="ペイロード重量（パイロット＋機構）[kg]")
    p.add_argument("--w-spar-init",  type=float, default=DEFAULT_W_INITIAL_SPAR,
                   help="桁重量初期推定値（片翼）[kg]")
    p.add_argument("--relax",        type=float, default=RELAX_ALPHA,
                   help="Picard 緩和係数 α")
    p.add_argument("--max-iter",     type=int,   default=MAX_ITER,
                   help="最大反復回数")
    p.add_argument("--conv-tol",     type=float, default=CONV_TOL,
                   help="収束閾値 [kg]")
    p.add_argument("--verbose",      action="store_true",
                   help="各反復の詳細を表示")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    print("=" * 70)
    print("  Iris — HPA 主翼スパー最適設計")
    print("=" * 70)
    print(f"  スパン        : {args.span:.0f} mm")
    print(f"  翼根コード長  : {args.chord:.0f} mm")
    print(f"  たわみ許容    : {args.delta_max:.0f} mm")
    print(f"  外皮面密度    : {args.rho_skin:.4f} kg/m²")
    print(f"  飛行速度      : {args.v:.1f} m/s")
    print(f"  空気密度      : {args.rho_air:.4f} kg/m³")
    print(f"  β（TR-797）  : {args.beta:.2f}")
    print(f"  ペイロード    : {args.payload:.1f} kg")
    print()

    result = run_convergence_loop(
        span_mm        = args.span,
        root_chord_mm  = args.chord,
        delta_max_mm   = args.delta_max,
        rho_skin_kg_m2 = args.rho_skin,
        v_ms           = args.v,
        rho_air        = args.rho_air,
        beta           = args.beta,
        payload_kg     = args.payload,
        w_spar_init_kg = args.w_spar_init,
        relax          = args.relax,
        max_iter       = args.max_iter,
        conv_tol       = args.conv_tol,
        verbose        = True,
    )

    result.print_summary()

    # 最終 EI 分布サマリー
    if result.EI_opt is not None:
        print(f"\n  EI 分布（翼根→翼端）")
        print(f"    翼根: {result.EI_opt[0]:.3e} N·mm²")
        print(f"    中間: {result.EI_opt[len(result.EI_opt)//2]:.3e} N·mm²")
        print(f"    翼端: {result.EI_opt[-1]:.3e} N·mm²")

    return 0 if result.converged else 1


if __name__ == "__main__":
    sys.exit(main())
