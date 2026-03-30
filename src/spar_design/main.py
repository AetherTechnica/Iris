"""
main.py — Iris スパー最適設計メインスクリプト
================================================
起動するとチャット形式で諸元を入力できる。
Enter のみで各項目のデフォルト値を使用する。

【実行例】
  python -m src.spar_design.main
  python main.py
"""

from __future__ import annotations

import sys

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


def _ask(prompt: str, default: float, fmt: str = ".4g") -> float:
    """
    1項目ぶんの入力を受け取る。
    Enter のみ → default を使用。
    不正な値が入力された場合は再入力を促す。
    """
    default_str = format(default, fmt)
    while True:
        try:
            raw = input(f"  {prompt} [{default_str}]: ").strip()
            if raw == "":
                return default
            return float(raw)
        except ValueError:
            print("    ※ 数値を入力してください。")


def _ask_yn(prompt: str, default: bool = True) -> bool:
    """y/n の入力を受け取る。"""
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"  {prompt} {suffix}: ").strip().lower()
    if raw == "":
        return default
    return raw.startswith("y")


def _input_params() -> dict:
    """チャット形式で諸元を入力し，パラメータ辞書を返す。"""
    print()
    print("  Enter のみでデフォルト値を使用します。")
    print()

    # --- 主要諸元 ---
    print("  【機体諸元】")
    span         = _ask("片翼スパン                  [mm]", DEFAULT_SPAN,        ".0f")
    chord        = _ask("翼根コード長                [mm]", DEFAULT_ROOT_CHORD,  ".0f")
    payload      = _ask("ペイロード（パイロット+機構）[kg]", DEFAULT_PAYLOAD,    ".1f")

    print()
    print("  【構造諸元】")
    delta_max    = _ask("翼端たわみ許容値            [mm]", DEFAULT_DELTA_MAX,   ".0f")
    rho_skin     = _ask("外皮面密度                  [kg/m²]", DEFAULT_RHO_SKIN, ".4f")

    print()
    print("  【空力・環境諸元】")
    v_ms         = _ask("飛行速度                    [m/s]", 7.5,               ".2f")
    rho_air      = _ask("空気密度                    [kg/m³]", 1.154,           ".4f")
    beta         = _ask("循環分布係数 β（TR-797）   [-]",  DEFAULT_BETA,        ".2f")

    print()
    print("  【計算設定】（通常はそのままでOK）")
    w_spar_init  = _ask("桁重量初期推定値（片翼）    [kg]", DEFAULT_W_INITIAL_SPAR, ".1f")
    relax        = _ask("Picard 緩和係数 α          [-]",  RELAX_ALPHA,         ".2f")
    max_iter     = int(_ask("最大反復回数               [-]",  MAX_ITER,        ".0f"))
    conv_tol     = _ask("収束閾値                    [kg]", CONV_TOL,           ".4f")

    return dict(
        span_mm        = span,
        root_chord_mm  = chord,
        delta_max_mm   = delta_max,
        rho_skin_kg_m2 = rho_skin,
        v_ms           = v_ms,
        rho_air        = rho_air,
        beta           = beta,
        payload_kg     = payload,
        w_spar_init_kg = w_spar_init,
        relax          = relax,
        max_iter       = max_iter,
        conv_tol       = conv_tol,
    )


def main() -> int:
    print("=" * 70)
    print("  Iris — HPA 主翼スパー最適設計")
    print("=" * 70)

    params = _input_params()

    # 確認表示
    print()
    print("  【入力諸元の確認】")
    print(f"    片翼スパン        : {params['span_mm']:.0f} mm")
    print(f"    翼根コード長      : {params['root_chord_mm']:.0f} mm")
    print(f"    ペイロード        : {params['payload_kg']:.1f} kg")
    print(f"    たわみ許容        : {params['delta_max_mm']:.0f} mm")
    print(f"    外皮面密度        : {params['rho_skin_kg_m2']:.4f} kg/m²")
    print(f"    飛行速度          : {params['v_ms']:.2f} m/s")
    print(f"    空気密度          : {params['rho_air']:.4f} kg/m³")
    print(f"    β                : {params['beta']:.2f}")
    print()

    ok = _ask_yn("この諸元で計算を開始しますか？", default=True)
    if not ok:
        print("  キャンセルしました。")
        return 0

    result = run_convergence_loop(**params, verbose=True)

    result.print_summary()

    if result.EI_opt is not None:
        print(f"\n  EI 分布（翼根→翼端）")
        print(f"    翼根: {result.EI_opt[0]:.3e} N·mm²")
        print(f"    中間: {result.EI_opt[len(result.EI_opt)//2]:.3e} N·mm²")
        print(f"    翼端: {result.EI_opt[-1]:.3e} N·mm²")

    return 0 if result.converged else 1


if __name__ == "__main__":
    sys.exit(main())
