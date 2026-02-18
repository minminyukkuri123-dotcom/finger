"""
ステップ4: カスタムデータ収集 & 追加学習ツール
目的: カメラで自分の手のポーズを撮影し、追加学習でモデルの精度を上げる。

使い方:
  source .venv/bin/activate
  python step4_collect_and_finetune.py

  === データ収集モード ===
  カメラが起動したら、手のポーズを見せて以下のキーを押す：
    'r' → グー（ROCK）として保存
    'p' → パー（PAPER）として保存
    's' → チョキ（SCISSORS）として保存
    'c' → 収集終了 → 追加学習を開始
    'q' → 全終了（学習しない）

  === 追加学習モード ===
  収集した画像で既存モデルを追加学習（fine-tuning）する。
  完了後、改良モデルで step3_realtime.py を実行すれば精度向上を確認できる。
"""

import os
import time
import glob
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- 設定の読み込み ---
from config import IMG_SIZE, RPS_MODEL_PATH, RPS_CLASSES

# MediaPipe Hand Landmarker モデルパス
HAND_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

# カスタムデータ保存先
CUSTOM_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "custom")

# キーとクラスの対応
KEY_CLASS_MAP = {
    ord('r'): 0,  # rock
    ord('p'): 1,  # paper
    ord('s'): 2,  # scissors
}


class HandCropper:
    """手の検出 & 背景除去 crop（step3と同じロジック）"""

    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def crop_hand(self, frame):
        """手を検出して背景除去cropを返す。検出なしならNone。"""
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return None, None

        hand_lm = result.hand_landmarks[0]
        pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_lm])

        x_min_raw, y_min_raw = pts.min(axis=0)
        x_max_raw, y_max_raw = pts.max(axis=0)
        hand_w = x_max_raw - x_min_raw
        hand_h = y_max_raw - y_min_raw
        margin = int(max(hand_w, hand_h) * 0.15)

        x_min = max(0, x_min_raw - margin)
        y_min = max(0, y_min_raw - margin)
        x_max = min(w, x_max_raw + margin)
        y_max = min(h, y_max_raw + margin)

        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        side = max(x_max - x_min, y_max - y_min)
        half = side // 2

        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)

        # 凸包マスクで背景を白に
        hull = cv2.convexHull(pts)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        kernel_size = max(15, int(max(hand_w, hand_h) * 0.15))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

        masked = np.full_like(frame, 255)
        masked[mask > 0] = frame[mask > 0]

        crop = masked[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None

        # リサイズして返す
        crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        bbox = (x1, y1, x2, y2)
        return crop_resized, bbox

    def release(self):
        self.detector.close()


def collect_data():
    """カメラでポーズを撮影してデータを収集する"""
    print("=" * 60)
    print("📸 データ収集モード")
    print("=" * 60)
    print("  キー操作:")
    print("    'r' → グー（ROCK）として保存")
    print("    'p' → パー（PAPER）として保存")
    print("    's' → チョキ（SCISSORS）として保存")
    print("    'c' → 収集終了 → 追加学習開始")
    print("    'q' → 全終了（学習しない）")
    print("=" * 60)

    # 保存ディレクトリ作成
    for cls in RPS_CLASSES:
        os.makedirs(os.path.join(CUSTOM_DATA_DIR, cls), exist_ok=True)

    cropper = HandCropper()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ カメラを開けませんでした。")
        return False

    # 各クラスの既存枚数を取得
    counts = {}
    for cls in RPS_CLASSES:
        d = os.path.join(CUSTOM_DATA_DIR, cls)
        counts[cls] = len(glob.glob(os.path.join(d, "*.png")))

    saved_this_session = {cls: 0 for cls in RPS_CLASSES}
    last_save_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()

            # 手を検出して crop プレビュー
            crop, bbox = cropper.crop_hand(frame)

            if crop is not None and bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # crop プレビューを右上に表示
                preview_size = 120
                preview = cv2.resize(crop, (preview_size, preview_size))
                display[10:10+preview_size, display.shape[1]-preview_size-10:display.shape[1]-10] = preview
                cv2.putText(display, "CROP", (display.shape[1]-preview_size-5, 10+preview_size+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(display, "No hand", (display.shape[1]//2-50, display.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

            # ヘッダー
            h, w, _ = display.shape
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

            cv2.putText(display, "DATA COLLECTION", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "[R]ock  [P]aper  [S]cissors  [C]omplete  [Q]uit",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 各クラスの枚数表示
            y_offset = h - 80
            for i, cls in enumerate(RPS_CLASSES):
                total = counts[cls] + saved_this_session[cls]
                text = f"{cls}: {total} pics (+{saved_this_session[cls]} new)"
                color = (0, 255, 0) if saved_this_session[cls] > 0 else (200, 200, 200)
                cv2.putText(display, text, (10, y_offset + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

            cv2.imshow('Data Collection', display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("⏹ 終了。学習しません。")
                cap.release()
                cv2.destroyAllWindows()
                cropper.release()
                return False

            if key == ord('c'):
                print("✅ データ収集完了！")
                break

            if key in KEY_CLASS_MAP and crop is not None:
                # 連打防止（0.2秒間隔）
                now = time.time()
                if now - last_save_time < 0.2:
                    continue
                last_save_time = now

                cls_idx = KEY_CLASS_MAP[key]
                cls_name = RPS_CLASSES[cls_idx]
                save_dir = os.path.join(CUSTOM_DATA_DIR, cls_name)

                # ファイル名をタイムスタンプで
                filename = f"{cls_name}_{int(now*1000)}.png"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, crop)

                saved_this_session[cls_name] += 1
                total = counts[cls_name] + saved_this_session[cls_name]
                print(f"  💾 {cls_name.upper()} を保存 ({total}枚目) → {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        cropper.release()

    # 今回保存した枚数を表示
    print("\n📊 今回の収集結果:")
    total_new = 0
    for cls in RPS_CLASSES:
        n = saved_this_session[cls]
        total_new += n
        total = counts[cls] + n
        print(f"  {cls}: +{n} 枚 (合計 {total} 枚)")

    if total_new == 0:
        print("⚠️ 新しい画像がありません。追加学習をスキップします。")
        return False

    return True


def load_custom_data():
    """カスタムデータを読み込む"""
    images = []
    labels = []

    for cls_idx, cls_name in enumerate(RPS_CLASSES):
        cls_dir = os.path.join(CUSTOM_DATA_DIR, cls_name)
        if not os.path.exists(cls_dir):
            continue

        files = glob.glob(os.path.join(cls_dir, "*.png"))
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(cls_idx)

    if not images:
        return None, None

    images = np.array(images, dtype=np.float32)
    images = preprocess_input(images)
    labels = np.array(labels, dtype=np.int32)

    return images, labels


def finetune_model():
    """既存モデルをカスタムデータで追加学習する"""
    print("\n" + "=" * 60)
    print("🏋️ 追加学習開始")
    print("=" * 60)

    # カスタムデータを読み込み
    x_custom, y_custom = load_custom_data()
    if x_custom is None:
        print("❌ カスタムデータがありません。")
        return

    print(f"📦 カスタムデータ: {len(x_custom)} 枚")
    for cls_idx, cls_name in enumerate(RPS_CLASSES):
        n = np.sum(y_custom == cls_idx)
        print(f"  {cls_name}: {n} 枚")

    # 既存モデルを読み込み
    print(f"\n📂 既存モデルを読み込み: {RPS_MODEL_PATH}")
    model = keras.models.load_model(RPS_MODEL_PATH)

    # バックボーン（MobileNetV2）の最後の数層も学習可能にする（fine-tuning）
    base_model = model.layers[0]  # Sequential の最初の層が MobileNetV2
    base_model.trainable = True

    # 最後の30層だけ学習可能にする
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"🔓 学習可能な層: {trainable_count} / {len(base_model.layers)}")

    # 低い学習率でコンパイル（fine-tuning用）
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # データシャッフル
    indices = np.random.permutation(len(x_custom))
    x_custom = x_custom[indices]
    y_custom = y_custom[indices]

    # データ拡張付きで学習（少データでも効果的に）
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
    )

    # 追加学習（少エポック＋早期終了）
    print("\n🔄 追加学習中...")
    history = model.fit(
        datagen.flow(x_custom, y_custom, batch_size=min(16, len(x_custom))),
        epochs=10,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='accuracy',
                patience=3,
                restore_best_weights=True,
                verbose=1
            )
        ]
    )

    final_acc = history.history['accuracy'][-1]
    print(f"\n📈 最終学習精度: {final_acc*100:.1f}%")

    # モデル保存（元のモデルを上書き）
    model.save(RPS_MODEL_PATH)
    print(f"💾 モデルを上書き保存: {RPS_MODEL_PATH}")

    # バックアップも保存
    backup_path = RPS_MODEL_PATH.replace(".keras", "_backup.keras")
    print(f"📋 (元のモデルのバックアップが必要な場合は事前にコピーしてください)")

    print("\n" + "=" * 60)
    print("✅ 追加学習完了！")
    print("   step3_realtime.py を再起動すれば改善されたモデルが使われます。")
    print("=" * 60)


import sys

def main():
    print("🚀 ステップ4: カスタムデータ収集 & 追加学習")
    print("=" * 60)

    # 引数チェック
    if "--train-only" in sys.argv:
        print("⏩ データ収集をスキップして追加学習のみ実行します。")
        finetune_model()
        return

    # 1. データ収集
    should_train = collect_data()

    # 2. 追加学習
    if should_train:
        finetune_model()


if __name__ == "__main__":
    main()
