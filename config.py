import os

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

# モデル保存先の絶対パス（実行環境に合わせて調整）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_model_path(filename):
    path = os.path.join(BASE_DIR, "models", filename)
    print(f"Checking path: {path}")
    print(f"Exists: {os.path.exists(path)}")
    return path

MNIST_MODEL_PATH = get_model_path("mnist_cnn.keras")
RPS_MODEL_PATH = get_model_path("rps_mobilenet.keras")
GS_MODEL_PATH = get_model_path("gs_mobilenet.keras")
HAND_MODEL_PATH = get_model_path("hand_landmarker.task")

# --- データディレクトリ ---
DATA_DIR = "data"
MNIST_DATA_DIR = "data/mnist"
RPS_DATA_DIR = "data/rps"
GS_DATA_DIR = "data/gs"

# --- クラスラベル（RPS） ---
RPS_CLASSES = ["rock", "paper", "scissors"]
RPS_LABELS_JP = ["✊ グー", "🖐 パー", "✌ チョキ"]

# --- クラスラベル（Gang Signs / Custom） ---
GS_CLASSES = ["crip", "westside", "piru", "killaz", "eastside"]

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
