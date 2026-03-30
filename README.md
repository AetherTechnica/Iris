# Iris — HPA主翼スパー最適設計プログラム

**プロジェクト**: AetherTechnica
**担当**: 和田慎之助
**目的**: B2機体の主翼スパーを物理ベースで最適設計するPythonプログラム

---

## 概要

人力飛行機（HPA）の主翼スパーについて，以下を自動で解く：

1. **LayupOptimizer** — 純物理ベース貪欲法で最小積層数を探索（AIサロゲート不要）
2. **MandrelDP** — 動的計画法で離散マンドレルの最適配置（かんざし継手コスト込み）
3. **収束ループ** — 自重による荷重増加を正確に追跡しながら全備重量を収束

Hemere（前プロジェクト）のコードは一切変更しない。

---

## 動作環境

```
Python    : anaconda3/envs/HPA
numpy     : 1.26.4
scipy     : 1.15.3
matplotlib: 3.10.7
```

---

## ディレクトリ構成

```
Iris/
├── src/spar_design/
│   ├── config.py            # 設計定数・デフォルトパラメータ
│   ├── mandrel.py           # Mandrelデータクラス＋10種リスト
│   ├── spar_calculator.py   # CFRP断面剛性・重量計算
│   ├── layup_optimizer.py   # 純物理貪欲法積層最適化  ← Phase 1 ✓
│   ├── load_calculator.py   # 荷重・曲げモーメント計算  ← Phase 3
│   ├── deflection_calc.py   # たわみ計算               ← Phase 3
│   ├── secondary_weight.py  # 二次構造重量（リブ・外皮）← Phase 3
│   ├── mandrel_dp.py        # DPマンドレル最適配置     ← Phase 2
│   ├── stiffness_optimizer.py # たわみ制約下の剛性最適化 ← Phase 3
│   ├── convergence_loop.py  # 重量推算収束ループ        ← Phase 4
│   └── main.py              # エントリーポイント        ← Phase 4
└── tests/spar_design/
```

---

## 単位系

| 量       | 単位     | 備考 |
|----------|----------|------|
| 長さ     | mm       | スパン，マンドレル径，たわみ全て |
| 曲げ剛性 | N·mm²    | SparCalculator内部は kgf·mm²，境界で変換 |
| 重量     | kg       | 線密度は kg/m |
| 荷重     | N/mm     | 分布荷重 |

---

## 実装状況

- [x] Phase 1: 基盤モジュール（config, mandrel, spar_calculator, layup_optimizer）
- [x] Phase 2: DPマンドレル最適配置（mandrel_dp.py）
- [ ] Phase 3: 最適化（たわみ・剛性・二次構造）
- [ ] Phase 4: 収束ループ統合
