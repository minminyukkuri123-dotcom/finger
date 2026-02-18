# 🤟 Finger Pose Detection - 静止ポーズ発動システム

カメラ映像からリアルタイムで**グー / チョキ / パー**を判定し、静止ポーズを検出して演出を発動するシステム。

## 概要

```
カメラ → MediaPipe手検出 → crop → MobileNetV2推論 → 発動判定 → 演出
```

| 演出 | ポーズ | 発動テキスト |
|------|--------|-------------|
| ✊ | グー | 最初はグー |
| 🖐 | パー | 結界展開 |
| ✌ | チョキ | 術式発動 |

## セットアップ

```bash
# 仮想環境を有効化
source .venv/bin/activate

# パッケージインストール（初回のみ）
pip install -r requirements.txt
```

## 使い方

### ステップ1: パイプライン確認（省略可）

学習コードの型作り。Fashion MNIST で CNN を学習。

```bash
MPLCONFIGDIR=/tmp/mpl_config python step1_mnist_train.py
```

### ステップ2: RPS モデル学習

MobileNetV2 の転移学習で「グー/チョキ/パー」分類モデルを構築。

```bash
MPLCONFIGDIR=/tmp/mpl_config python step2_rps_train.py
```

学習結果:
- テスト精度: 約84%
- 推論速度: 約63ms/フレーム
- モデル保存先: `models/rps_mobilenet.keras`

### ステップ3: リアルタイム推論

カメラで手のポーズをリアルタイム判定。

```bash
python step3_realtime.py
```

操作:
- カメラに手をかざしてポーズを見せる
- 同じポーズを数秒キープ → 演出が発動！
- `q` キーで終了

## 発動条件

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| 確信度閾値 | 0.80 | この確率を超えた予測のみ有効 |
| 連続フレーム数 | 8 | 同一ポーズがこのフレーム数連続で検出 |
| クールダウン | 1.0秒 | 発動後の再発動禁止時間 |

パラメータは `config.py` で変更可能。

## ファイル構成

```
finger/
├── config.py               # 設定値（閾値・フレーム数等）
├── requirements.txt         # 依存パッケージ
├── step1_mnist_train.py     # ステップ1: MNIST学習
├── step2_rps_train.py       # ステップ2: RPS転移学習
├── step3_realtime.py        # ステップ3: リアルタイム推論
├── models/                  # 学習済みモデル
├── data/                    # データセット
└── .venv/                   # Python仮想環境
```

## 拡張方法

1. **自分の印を追加**: カメラで自分の手を50〜200枚/クラス撮影し、`step2_rps_train.py` を参考に追加学習
2. **演出の変更**: `config.py` の `TRIGGER_EFFECTS` を編集
3. **感度調整**: `config.py` の `CONFIDENCE_THRESHOLD`、`TRIGGER_FRAMES`、`COOLDOWN_SEC` を調整

## 技術スタック

- **TensorFlow 2.20** + **MobileNetV2**（転移学習）
- **MediaPipe Hands**（手領域検出）
- **OpenCV**（カメラ映像・UI表示）
