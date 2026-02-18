"""
ステップ3: リアルタイム推論 + 発動判定
目的: カメラ映像から手のグー/チョキ/パーをリアルタイム判定し、
      静止ポーズを検出した際に演出を発動する。

使い方:
  source .venv/bin/activate
  python step3_realtime.py
  - カメラに手をかざしてポーズを見せる
  - 同じポーズを数秒キープ → 演出が発動
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
    IMG_SIZE, RPS_MODEL_PATH, RPS_CLASSES, RPS_LABELS_JP,
    CONFIDENCE_THRESHOLD, TRIGGER_FRAMES, COOLDOWN_SEC,
    TRIGGER_EFFECTS
)

# MediaPipe Hand Landmarker モデルパス
HAND_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")


class HandDetector:
    """MediaPipe HandLandmarker による手の検出・切り出し"""

    def __init__(self, model_path=HAND_MODEL_PATH, num_hands=1,
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
        手を検出し、背景を除去したcrop画像を返す。
        ランドマークの凸包でマスクを作り、背景を白で塗りつぶす。
        戻り値: (crop_image, bbox, landmarks_list) or (None, None, None)
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return None, None, None

        # 最初に検出された手を使用
        hand_lm = result.hand_landmarks[0]

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
        # 手の凸包マスクを作成（やや膨張させて手全体をカバー）
        hull = cv2.convexHull(pts)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        # マスクを膨張（手の輪郭からはみ出す部分もカバー）
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
            return None, None, None

        bbox = (x1, y1, x2, y2)
        return crop, bbox, hand_lm

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


class TriggerLogic:
    """静止ポーズ発動判定ロジック"""

    def __init__(self, trigger_frames=TRIGGER_FRAMES,
                 confidence_threshold=CONFIDENCE_THRESHOLD,
                 cooldown_sec=COOLDOWN_SEC):
        self.trigger_frames = trigger_frames
        self.confidence_threshold = confidence_threshold
        self.cooldown_sec = cooldown_sec

        self.buffer = collections.deque(maxlen=trigger_frames)
        self.next_allowed_time = 0.0
        self.last_triggered = None
        self.trigger_display_time = 0.0

    def update(self, pred_class, confidence):
        """
        フレームの予測結果を追加し、発動条件を判定する。
        戻り値: 発動されたクラス（発動しなければ None）
        """
        self.buffer.append(pred_class)

        now = time.time()

        if (confidence > self.confidence_threshold and
            len(self.buffer) >= self.trigger_frames and
            all(b == self.buffer[-1] for b in self.buffer) and
            now > self.next_allowed_time):

            triggered_class = pred_class
            self.next_allowed_time = now + self.cooldown_sec
            self.last_triggered = triggered_class
            self.trigger_display_time = now
            return triggered_class

        return None

    def get_display_effect(self):
        """現在表示すべき演出テキストを返す（発動後2秒間表示）"""
        if self.last_triggered is not None:
            elapsed = time.time() - self.trigger_display_time
            if elapsed < 2.0:
                return TRIGGER_EFFECTS.get(self.last_triggered, "")
        return None

    def is_in_cooldown(self):
        """クールダウン中かどうか"""
        return time.time() < self.next_allowed_time


def draw_ui(frame, bbox, pred_class, confidence, probs,
            trigger_logic, fps, hand_detected):
    """UIを描画"""
    h, w, _ = frame.shape

    # 上部の情報バー
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS表示
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # クールダウン表示
    if trigger_logic.is_in_cooldown():
        remaining = trigger_logic.next_allowed_time - time.time()
        cv2.putText(frame, f"COOLDOWN: {remaining:.1f}s",
                    (w - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    if not hand_detected:
        cv2.putText(frame, "No hand detected", (w//2 - 100, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        return frame

    # バウンディングボックス
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # クラス名と確信度
    label = f"{RPS_CLASSES[pred_class].upper()} ({confidence*100:.0f}%)"
    cv2.putText(frame, label, (w//2 - 80, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 確率バー
    bar_x = 10
    bar_y = h - 100
    for i, (cls, prob) in enumerate(zip(RPS_CLASSES, probs)):
        bar_w = int(prob * 200)
        color = (0, 255, 0) if i == pred_class else (100, 100, 100)
        cv2.rectangle(frame, (bar_x, bar_y + i*25),
                      (bar_x + bar_w, bar_y + i*25 + 18), color, -1)
        cv2.putText(frame, f"{cls}: {prob*100:.0f}%",
                    (bar_x + 5, bar_y + i*25 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # バッファ進捗バー（発動までの進捗）
    if confidence > CONFIDENCE_THRESHOLD:
        same_count = 0
        for b in reversed(trigger_logic.buffer):
            if b == pred_class:
                same_count += 1
            else:
                break
        progress = min(same_count / TRIGGER_FRAMES, 1.0)
        bar_w = int(progress * (w - 20))
        cv2.rectangle(frame, (10, h - 15), (10 + bar_w, h - 5),
                      (0, 255, 255), -1)
        cv2.rectangle(frame, (10, h - 15), (w - 10, h - 5),
                      (100, 100, 100), 1)

    # 発動演出
    effect = trigger_logic.get_display_effect()
    if effect is not None:
        elapsed = time.time() - trigger_logic.trigger_display_time
        alpha = max(0, 1.0 - elapsed / 2.0)

        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h//2 - 60), (w, h//2 + 60),
                      (0, 0, 128), -1)
        cv2.addWeighted(overlay2, 0.5 * alpha, frame, 1 - 0.5 * alpha, 0, frame)

        effect_texts = {
            0: "ROCK - First Strike!",
            1: "PAPER - Barrier Expand!",
            2: "SCISSORS - Technique Activate!"
        }
        text = effect_texts.get(trigger_logic.last_triggered, "TRIGGERED!")
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, text, (text_x, h//2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 255), 3)

    return frame


def main():
    """メイン処理"""
    print("🚀 ステップ3: リアルタイム推論 + 発動判定")
    print("=" * 60)

    # モデル読み込み
    if not os.path.exists(RPS_MODEL_PATH):
        print(f"❌ RPS モデルが見つかりません: {RPS_MODEL_PATH}")
        print("   先に step2_rps_train.py を実行してください。")
        return

    if not os.path.exists(HAND_MODEL_PATH):
        print(f"❌ HandLandmarker モデルが見つかりません: {HAND_MODEL_PATH}")
        print("   models/hand_landmarker.task をダウンロードしてください。")
        return

    classifier = PoseClassifier(RPS_MODEL_PATH)
    hand_detector = HandDetector()
    trigger_logic = TriggerLogic()

    # カメラ起動
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ カメラを開けませんでした。")
        return

    print("\n📷 カメラ起動！")
    print("   手をカメラにかざしてグー/チョキ/パーを判定します")
    print("   同じポーズをキープ → 演出が発動！")
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

            # 手の検出
            crop, bbox, landmarks = hand_detector.detect(frame)

            if crop is not None:
                # ポーズ分類
                pred_class, confidence, probs = classifier.predict(crop)

                # 発動判定
                triggered = trigger_logic.update(pred_class, confidence)

                if triggered is not None:
                    trigger_name = RPS_CLASSES[triggered]
                    print(f"🎯 発動！ {trigger_name.upper()} "
                          f"({TRIGGER_EFFECTS.get(triggered, '')})")

                # ランドマーク描画
                hand_detector.draw_landmarks(frame, landmarks)

                # UI描画
                frame = draw_ui(frame, bbox, pred_class, confidence,
                                probs, trigger_logic, fps, True)
            else:
                frame = draw_ui(frame, None, 0, 0.0, [0, 0, 0],
                                trigger_logic, fps, False)

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
