# ===================================
# 設定ファイル（ハイパーパラメータ）
# ===================================

# --- 画像サイズ ---
IMG_SIZE = 224  # MobileNetV2 入力サイズ
MNIST_IMG_SIZE = 28  # Sign Language MNIST 入力サイズ

# --- 学習パラメータ ---
BATCH_SIZE = 32
EPOCHS_MNIST = 10  # ステップ1: MNIST 学習エポック数
EPOCHS_RPS = 15    # ステップ2: RPS 学習エポック数
LEARNING_RATE = 0.001

# --- モデル保存先 ---
MNIST_MODEL_PATH = "models/mnist_cnn.keras"
RPS_MODEL_PATH = "models/rps_mobilenet.keras"

# --- データディレクトリ ---
DATA_DIR = "data"
MNIST_DATA_DIR = "data/mnist"
RPS_DATA_DIR = "data/rps"

# --- クラスラベル（RPS） ---
RPS_CLASSES = ["rock", "paper", "scissors"]
RPS_LABELS_JP = ["✊ グー", "🖐 パー", "✌ チョキ"]

# --- 発動判定パラメータ ---
CONFIDENCE_THRESHOLD = 0.80  # 確信度の閾値
TRIGGER_FRAMES = 8            # 連続フレーム数
COOLDOWN_SEC = 1.0            # クールダウン秒数

# --- 演出テキスト ---
TRIGGER_EFFECTS = {
    0: "✊ 最初はグー",
    1: "🖐 結界展開",
    2: "✌ 術式発動",
}
