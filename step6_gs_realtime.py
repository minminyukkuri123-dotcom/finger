"""
ステップ6: 新ポーズ（GSシリーズ）リアルタイム判定
目的: 学習した新しいポーズ（crip, westside, piru, killaz, eastside）を
      カメラ映像からリアルタイムで判定する（最大2つの手に対応）。
      検出率90%超えでサングラスが降ってくるアニメーション付き！

使い方:
  source .venv/bin/activate
  python step6_gs_realtime.py
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
    IMG_SIZE, GS_MODEL_PATH, GS_CLASSES,
    CONFIDENCE_THRESHOLD, HAND_MODEL_PATH
)


def create_pixel_sunglasses(width=200, height=60):
    """ピクセルアートのサングラス画像を生成 (BGRA, 透過付き)"""
    # 高解像度のテンプレートを作成して後でリサイズ
    grid_w, grid_h = 40, 12
    pixel_size = 5  # 1ピクセルの実サイズ
    img_w = grid_w * pixel_size
    img_h = grid_h * pixel_size
    img = np.zeros((img_h, img_w, 4), dtype=np.uint8)  # BGRA, 全透明

    # サングラスのピクセルパターン (1=黒, 2=白ハイライト, 0=透明)
    pattern = [
        #  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
        [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],  # 0
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],  # 1
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],  # 2
        [0,1,1,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,0],  # 3
        [1,1,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],  # 4
        [1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],  # 5
        [1,1,1,1,1,1,1,1,1,2,2,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,2,2,1,1,1,1,1,1,1,1,1],  # 6
        [1,1,1,1,1,1,1,1,2,2,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,2,2,1,1,1,1,1,1,1,1,1,1],  # 7
        [0,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,0],  # 8
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],  # 9
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],  # 10
        [0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],  # 11
    ]

    for row_idx, row in enumerate(pattern):
        for col_idx, val in enumerate(row):
            if val == 0:
                continue
            x = col_idx * pixel_size
            y = row_idx * pixel_size
            if val == 1:
                color = (0, 0, 0, 255)  # 黒
            elif val == 2:
                color = (255, 255, 255, 255)  # 白ハイライト
            img[y:y+pixel_size, x:x+pixel_size] = color

    # リサイズ
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    return img


def overlay_transparent(bg, fg, x, y):
    """透過画像をBGR画像に合成"""
    h, w = fg.shape[:2]
    bh, bw = bg.shape[:2]

    # 画面外クリッピング
    if x >= bw or y >= bh or x + w <= 0 or y + h <= 0:
        return bg

    # クリッピング計算
    src_x1 = max(0, -x)
    src_y1 = max(0, -y)
    src_x2 = min(w, bw - x)
    src_y2 = min(h, bh - y)

    dst_x1 = max(0, x)
    dst_y1 = max(0, y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return bg

    fg_crop = fg[src_y1:src_y2, src_x1:src_x2]
    alpha = fg_crop[:, :, 3:4].astype(float) / 255.0
    fg_rgb = fg_crop[:, :, :3].astype(float)
    bg_region = bg[dst_y1:dst_y2, dst_x1:dst_x2].astype(float)

    bg[dst_y1:dst_y2, dst_x1:dst_x2] = (
        fg_rgb * alpha + bg_region * (1 - alpha)
    ).astype(np.uint8)

    return bg


class SunglassesAnimation:
    """サングラスが上から降ってくるアニメーション"""

    def __init__(self):
        self.active = False
        self.start_time = 0.0
        self.target_x = 0
        self.target_y = 0
        self.sg_width = 0
        self.sg_height = 0
        self.cooldown_until = 0.0
        self.drop_duration = 0.5  # 降下時間（秒）
        self.stay_duration = 3.0  # 装着表示時間（秒）
        self.smooth_x = 0.0
        self.smooth_y = 0.0

    def trigger(self, face_cx, face_cy, face_width):
        """サングラスアニメーション発動"""
        now = time.time()
        if now < self.cooldown_until:
            return
        if self.active:
            return

        self.active = True
        self.start_time = now
        self.sg_width = int(face_width * 1.1)
        self.sg_height = int(self.sg_width * 0.3)
        self.target_x = face_cx - self.sg_width // 2
        self.target_y = face_cy - self.sg_height // 2
        self.smooth_x = float(self.target_x)
        self.smooth_y = float(self.target_y)

    def update_tracking(self, face_cx, face_cy, face_width):
        """装着中に顔の位置を追従する"""
        if not self.active:
            return
        elapsed = time.time() - self.start_time
        if elapsed < self.drop_duration:
            return  # 降下中はトラッキングしない

        # サイズも更新
        self.sg_width = int(face_width * 1.1)
        self.sg_height = int(self.sg_width * 0.3)

        new_x = face_cx - self.sg_width // 2
        new_y = face_cy - self.sg_height // 2

        # スムージング（滑らかに追従）
        alpha = 0.35
        self.smooth_x = self.smooth_x * (1 - alpha) + new_x * alpha
        self.smooth_y = self.smooth_y * (1 - alpha) + new_y * alpha
        self.target_x = int(self.smooth_x)
        self.target_y = int(self.smooth_y)

    def draw(self, frame):
        """アニメーションを描画"""
        if not self.active:
            return frame

        now = time.time()
        elapsed = now - self.start_time
        total_duration = self.drop_duration + self.stay_duration

        if elapsed > total_duration:
            self.active = False
            self.cooldown_until = now + 2.0  # 2秒クールダウン
            return frame

        # サングラス画像を生成
        sg_img = create_pixel_sunglasses(self.sg_width, self.sg_height)

        if elapsed < self.drop_duration:
            # 降下アニメーション（イージング付き）
            t = elapsed / self.drop_duration
            t = 1 - (1 - t) ** 3
            start_y = -self.sg_height - 50
            current_y = int(start_y + (self.target_y - start_y) * t)
            current_x = self.target_x
        else:
            # 装着状態（トラッキングされた位置を使用）
            current_x = self.target_x
            current_y = self.target_y
            # 微揺れ効果（着地直後のみ）
            shake_elapsed = elapsed - self.drop_duration
            if shake_elapsed < 0.15:
                current_y += int(3 * np.sin(shake_elapsed * 40))

        frame = overlay_transparent(frame, sg_img, current_x, current_y)

        # 「DEAL WITH IT」テキスト表示（装着後）
        if elapsed > self.drop_duration + 0.2:
            fade = min(1.0, (elapsed - self.drop_duration - 0.2) / 0.3)
            text = "DEAL WITH IT"
            h, w = frame.shape[:2]
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            tx = (w - text_size[0]) // 2
            ty = h - 40
            color = (0, int(255 * fade), int(255 * fade))
            cv2.putText(frame, text, (tx+2, ty+2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        return frame


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
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return []

        detection_results = []
        for hand_lm in result.hand_landmarks:
            pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_lm])
            x_min_raw, y_min_raw = pts.min(axis=0)
            x_max_raw, y_max_raw = pts.max(axis=0)
            hand_w, hand_h = x_max_raw - x_min_raw, y_max_raw - y_min_raw
            margin = int(max(hand_w, hand_h) * 0.15)

            x_min, y_min = max(0, x_min_raw - margin), max(0, y_min_raw - margin)
            x_max, y_max = min(w, x_max_raw + margin), min(h, y_max_raw + margin)

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

            masked_frame = np.full_like(frame, 255)
            masked_frame[mask > 0] = frame[mask > 0]

            crop = masked_frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            detection_results.append((crop, (x1, y1, x2, y2), hand_lm))

        return detection_results

    def draw_landmarks(self, frame, hand_lm):
        h, w, _ = frame.shape
        connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
        for s, e in connections:
            pt1 = (int(hand_lm[s].x * w), int(hand_lm[s].y * h))
            pt2 = (int(hand_lm[e].x * w), int(hand_lm[e].y * h))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        for lm in hand_lm:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (255, 0, 0), -1)

    def release(self):
        self.detector.close()


class PoseClassifier:
    def __init__(self, model_path):
        print(f"📂 モデル読み込み中: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        self.model.predict(dummy, verbose=0)

    def predict(self, crop_image):
        img = cv2.resize(crop_image, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img.astype(np.float32))
        img = np.expand_dims(img, axis=0)
        probs = self.model.predict(img, verbose=0)[0]
        return np.argmax(probs), float(np.max(probs))


def main():
    print("🚀 ステップ6: 新ポーズ判定 (両手対応 + サングラス)")
    print("=" * 60)

    classifier = PoseClassifier(GS_MODEL_PATH)
    hand_detector = HandDetector(num_hands=2)

    # 顔検出用 (OpenCV Haar Cascade)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # サングラスアニメーション
    sg_anim = SunglassesAnimation()

    cap = cv2.VideoCapture(0)

    print("\n📷 判定開始！ 'q'キーで終了")
    print("   90%超えでサングラスが降ってきます 🕶️")

    fps_counter = collections.deque(maxlen=30)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        now = time.time()
        fps_counter.append(now - prev_time)
        prev_time = now
        fps = 1.0 / (sum(fps_counter) / len(fps_counter)) if fps_counter else 0

        # 顔検出（サングラスの位置決め用）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # 手の検出
        detect_results = hand_detector.detect(frame)
        high_confidence = False

        for crop, bbox, landmarks in detect_results:
            pred_class, confidence = classifier.predict(crop)
            hand_detector.draw_landmarks(frame, landmarks)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{GS_CLASSES[pred_class].upper()} ({confidence*100:.0f}%)"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # 90%超えで発動
            if confidence > 0.90:
                high_confidence = True

        # サングラスアニメーション発動 & トラッキング
        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            face_cx = fx + fw // 2
            face_eye_y = fy + int(fh * 0.35)

            if high_confidence:
                sg_anim.trigger(face_cx, face_eye_y, fw)

            # 装着中は常に顔をトラッキング
            sg_anim.update_tracking(face_cx, face_eye_y, fw)

        # サングラス描画
        frame = sg_anim.draw(frame)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('GS Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    hand_detector.release()

if __name__ == "__main__":
    main()
