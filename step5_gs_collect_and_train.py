"""
ステップ5: 新しい手の形（GSシリーズ）データ収集 & 学習ツール
目的: crip, westside, piru, killaz, eastside のポーズを撮影し、
      新しいデータセットとしてモデルを作成する。

使い方:
  source .venv/bin/activate
  python step5_gs_collect_and_train.py

  キー操作:
    '1' → crip として保存
    '2' → westside として保存
    '3' → piru として保存
    '4' → killaz として保存
    '5' → eastside として保存
    'c' → 収集終了 → 学習を開始
    'q' → 全終了
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
from config import IMG_SIZE, BATCH_SIZE, GS_MODEL_PATH, GS_CLASSES, GS_DATA_DIR, HAND_MODEL_PATH

# キーとクラスの対応
KEY_CLASS_MAP = {
    ord('1'): 0,
    ord('2'): 1,
    ord('3'): 2,
    ord('4'): 3,
    ord('5'): 4,
}


class HandCropper:
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

        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
        side = max(x_max - x_min, y_max - y_min)
        half = side // 2

        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)

        hull = cv2.convexHull(pts)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        kernel_size = max(15, int(max(hand_w, hand_h) * 0.15))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

        masked = np.full_like(frame, 255)
        masked[mask > 0] = frame[mask > 0]

        crop = masked[y1:y2, x1:x2]
        if crop.size == 0: return None, None

        crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        return crop_resized, (x1, y1, x2, y2)

    def release(self):
        self.detector.close()


def collect_data():
    print("=" * 60)
    print("📸 新ポーズデータ収集モード")
    print("=" * 60)
    for i, cls in enumerate(GS_CLASSES):
        print(f"  [{i+1}] → {cls}")
    print("  [C] → 収集終了 & 学習開始")
    print("  [Q] → 中断")
    print("=" * 60)

    for cls in GS_CLASSES:
        os.makedirs(os.path.join(GS_DATA_DIR, cls), exist_ok=True)

    cropper = HandCropper()
    cap = cv2.VideoCapture(0)
    counts = {cls: len(glob.glob(os.path.join(GS_DATA_DIR, cls, "*.png"))) for cls in GS_CLASSES}
    saved_this_session = {cls: 0 for cls in GS_CLASSES}
    last_save_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            display = frame.copy()

            crop, bbox = cropper.crop_hand(frame)
            if crop is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                preview = cv2.resize(crop, (120, 120))
                display[10:130, display.shape[1]-130:display.shape[1]-10] = preview

            # UI Header
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (display.shape[1], 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
            cv2.putText(display, "GS DATA COLLECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Status
            for i, cls in enumerate(GS_CLASSES):
                txt = f"{i+1}:{cls} ({counts[cls] + saved_this_session[cls]})"
                cv2.putText(display, txt, (10, 100 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('GS Collection', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): return False
            if key == ord('c'): break
            if key in KEY_CLASS_MAP and crop is not None:
                now = time.time()
                if now - last_save_time < 0.2: continue
                last_save_time = now
                cls_name = GS_CLASSES[KEY_CLASS_MAP[key]]
                filepath = os.path.join(GS_DATA_DIR, cls_name, f"{cls_name}_{int(now*1000)}.png")
                cv2.imwrite(filepath, crop)
                saved_this_session[cls_name] += 1
                print(f"💾 {cls_name} saved.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cropper.release()
    return sum(saved_this_session.values()) > 0


def train_gs_model():
    print("\n🏋️ 新モデルの学習を開始中...")
    images, labels = [], []
    for idx, cls in enumerate(GS_CLASSES):
        files = glob.glob(os.path.join(GS_DATA_DIR, cls, "*.png"))
        print(f"  {cls}: {len(files)} 枚")
        for f in files:
            img = cv2.imread(f)
            if img is None: continue
            img = cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(idx)

    if not images:
        print("❌ データがありません。")
        return

    x = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"\n📦 合計: {len(x)} 枚, クラス数: {len(GS_CLASSES)}")

    # データをシャッフル
    indices = np.random.permutation(len(x))
    x = x[indices]
    y = y[indices]

    # 前処理
    x = preprocess_input(x)

    # バリデーション分割（20%）
    split = int(len(x) * 0.8)
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"📊 学習: {len(x_train)} 枚, 検証: {len(x_val)} 枚")

    # モデル構築
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet'
    )
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(GS_CLASSES), activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # データ拡張
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    # 学習（データ拡張あり + Early Stopping）
    print("\n🔄 学習中...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        epochs=30,
        validation_data=(x_val, y_val),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=5,
                restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, verbose=1
            )
        ],
        verbose=1
    )

    # 結果表示
    val_acc = history.history.get('val_accuracy', [0])[-1]
    print(f"\n📈 最終検証精度: {val_acc*100:.1f}%")

    os.makedirs(os.path.dirname(GS_MODEL_PATH), exist_ok=True)
    model.save(GS_MODEL_PATH)
    print(f"✅ モデルを保存しました: {GS_MODEL_PATH}")


import sys

if __name__ == "__main__":
    if "--train-only" in sys.argv:
        print("⏩ データ収集をスキップして学習のみ実行します。")
        train_gs_model()
    else:
        if collect_data():
            train_gs_model()
