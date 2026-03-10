"""
ステップ3: リアルタイム推論
目的: カメラ映像から手のグー/チョキ/パーをリアルタイム判定し、
      検出したポーズを表示する（最大2つの手に対応）。

使い方:
  source .venv/bin/activate
  python step3_realtime.py
  - カメラに手をかざしてポーズを見せる
  - 'q' キーで終了
"""

import os
import time
import collections
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- 設定の読み込み ---
from config import (
    IMG_SIZE, RPS_MODEL_PATH, RPS_CLASSES,
    CONFIDENCE_THRESHOLD, HAND_MODEL_PATH
)


class HandDetector:
    """MediaPipe HandLandmarker による手の検出・切り出し"""

    def __init__(self, model_path=HAND_MODEL_PATH, num_hands=2,
                 min_detection_confidence=0.5):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame):
        """
        [両手対応]
        手を検出し、背景を除去したcrop画像を返す。
        戻り値: results = [(crop_image, bbox, landmarks_list), ...]
        手が検出されなければ空リスト [] を返す。
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return []

        detection_results = []

        # 検出されたすべての手に対して処理
        for hand_lm in result.hand_landmarks:
            # ランドマーク座標をピクセル座標に変換
            pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_lm])

            # バウンディングボックスを計算（手のサイズに比例したマージン）
            x_min_raw = pts[:, 0].min()
            y_min_raw = pts[:, 1].min()
            x_max_raw = pts[:, 0].max()
            y_max_raw = pts[:, 1].max()

            hand_w = x_max_raw - x_min_raw
            hand_h = y_max_raw - y_min_raw
            margin = int(max(hand_w, hand_h) * 0.15)  # 手のサイズの15%分マージン

            x_min = max(0, x_min_raw - margin)
            y_min = max(0, y_min_raw - margin)
            x_max = min(w, x_max_raw + margin)
            y_max = min(h, y_max_raw + margin)

            # 正方形にパディング
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            side = max(x_max - x_min, y_max - y_min)
            half = side // 2

            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, cx + half)
            y2 = min(h, cy + half)

            # --- 背景を白で塗りつぶしたcrop画像を作成 ---
            # 手の凸包マスクを作成
            hull = cv2.convexHull(pts)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)

            # マスクを膨張
            kernel_size = max(15, int(max(hand_w, hand_h) * 0.15))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (kernel_size, kernel_size))
            mask = cv2.dilate(mask, kernel, iterations=1)

            # 白背景に手だけ合成
            masked_frame = np.full_like(frame, 255)  # 白背景
            masked_frame[mask > 0] = frame[mask > 0]

            # 切り出し
            crop = masked_frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            bbox = (x1, y1, x2, y2)
            detection_results.append((crop, bbox, hand_lm))

        return detection_results

    def draw_landmarks(self, frame, hand_lm):
        """手のランドマークを描画"""
        h, w, _ = frame.shape
        # ランドマーク接続線の定義
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]
        # 接続線を描画
        for start_idx, end_idx in connections:
            start = hand_lm[start_idx]
            end = hand_lm[end_idx]
            pt1 = (int(start.x * w), int(start.y * h))
            pt2 = (int(end.x * w), int(end.y * h))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # ランドマーク点を描画
        for lm in hand_lm:
            cx_lm = int(lm.x * w)
            cy_lm = int(lm.y * h)
            cv2.circle(frame, (cx_lm, cy_lm), 4, (255, 0, 0), -1)

    def release(self):
        """リソースを解放"""
        self.detector.close()


class PoseClassifier:
    """RPS学習済みモデルによるポーズ分類"""

    def __init__(self, model_path):
        print(f"📂 モデルを読み込み中: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("✅ モデル読み込み完了")

        # ウォームアップ推論
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        self.model.predict(dummy, verbose=0)

    def predict(self, crop_image):
        """
        crop画像からポーズを推論する。
        戻り値: (pred_class, confidence, probs)
        """
        img = cv2.resize(crop_image, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        probs = self.model.predict(img, verbose=0)[0]
        pred_class = np.argmax(probs)
        confidence = float(np.max(probs))

        return pred_class, confidence, probs


def draw_ui(frame, results, fps):
    """UIを描画 (複数手対応)"""
    h, w, _ = frame.shape

    # 上部の情報バー
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS表示
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if not results:
        cv2.putText(frame, "No hand detected", (w//2 - 100, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        return frame

    # 検出された各手に対して描画
    for i, res in enumerate(results):
        # res: (bbox, pred_class, confidence, probs)
        # ※ main関数内で detector.detect の戻り値とは別に、予測結果を含めた構造にする想定
        bbox, pred_class, confidence = res
        
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # バウンディングボックスの上にラベル表示
            label = f"{RPS_CLASSES[pred_class].upper()} ({confidence*100:.0f}%)"
            
            # テキスト背景
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), color, -1)
            
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


def main():
    """メイン処理"""
    print("🚀 ステップ3: リアルタイム推論 (両手対応)")
    print("=" * 60)

    # モデル読み込み
    classifier = PoseClassifier(RPS_MODEL_PATH)
    # num_hands=2 に変更して両手検出に対応
    hand_detector = HandDetector(num_hands=2)

    # カメラ起動
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ カメラを開けませんでした。")
        return

    print("\n📷 カメラ起動！")
    print("   手をカメラにかざしてグー/チョキ/パーを判定します (最大2つ)")
    print("   'q' キーで終了")
    print("=" * 60)

    fps_counter = collections.deque(maxlen=30)
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 左右反転（鏡像）
            frame = cv2.flip(frame, 1)

            # FPS計算
            now = time.time()
            fps_counter.append(now - prev_time)
            prev_time = now
            fps = 1.0 / (sum(fps_counter) / len(fps_counter)) if fps_counter else 0

            # 手の検出 (0〜2つの手が返ってくる)
            # detect_results: [(crop, bbox, landmarks), ...]
            detect_results = hand_detector.detect(frame)

            ui_results = []

            for crop, bbox, landmarks in detect_results:
                # ポーズ分類
                pred_class, confidence, probs = classifier.predict(crop)

                # UI表示用の情報を保存
                ui_results.append((bbox, pred_class, confidence))

                # ランドマーク描画
                hand_detector.draw_landmarks(frame, landmarks)
                
                # ターミナルへデバッグ出力（確信度が高い場合のみ）
                if confidence > CONFIDENCE_THRESHOLD:
                    pass 
                    # 必要であればここでprint出力

            # UI描画
            frame = draw_ui(frame, ui_results, fps)

            # 表示
            cv2.imshow('Finger Pose Detection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n⏹ 中断されました")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_detector.release()
        print("👋 終了しました")


if __name__ == "__main__":
    main()
