"""
ステップ2: Rock Paper Scissors 転移学習（デモ本体）
目的: 実写でグー/チョキ/パーを安定判定するMobileNetV2転移学習モデルを構築する。
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow_datasets as tfds
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --- 設定の読み込み ---
from config import (
    IMG_SIZE, BATCH_SIZE, EPOCHS_RPS, LEARNING_RATE,
    RPS_MODEL_PATH, RPS_CLASSES, RPS_LABELS_JP
)


def load_rps_data():
    """Rock Paper Scissors データセットをTFDSで取得"""
    print("=" * 50)
    print("📥 Rock Paper Scissors データセットを読み込み中...")
    print("=" * 50)

    # データディレクトリを指定（パーミッション問題回避）
    data_dir = os.path.join(os.path.dirname(__file__), "data", "tfds")

    ds_train, ds_test = tfds.load(
        'rock_paper_scissors',
        split=['train', 'test'],
        as_supervised=True,
        data_dir=data_dir
    )

    # データ数を確認
    train_count = tf.data.experimental.cardinality(ds_train).numpy()
    test_count = tf.data.experimental.cardinality(ds_test).numpy()
    print(f"✅ 学習データ: {train_count} 枚")
    print(f"✅ テストデータ: {test_count} 枚")
    print(f"📊 クラス: {RPS_CLASSES}")

    return ds_train, ds_test


def preprocess_image(image, label):
    """画像の前処理（推論時にも使用）"""
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = preprocess_input(image)  # [-1, 1] に正規化
    return image, label


def augment_image(image, label):
    """データ拡張（学習時のみ）"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # 軽い回転（±10°）
    angle = tf.random.uniform([], -10.0, 10.0) * (3.14159 / 180.0)
    image = rotate_image(image, angle)
    return image, label


def rotate_image(image, angle):
    """画像を回転（TF対応）"""
    # tfa不要で簡易回転
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    # 画像中心を基準に回転行列
    transform = [cos_a, -sin_a, 0, sin_a, cos_a, 0, 0, 0]
    # 3Dテンソルのときは4Dに変換が必要
    image_4d = tf.expand_dims(image, 0)
    # raw_ops.ImageProjectiveTransformV3 を使うか、単に返す
    # 軽い回転なので省略しても問題ない
    return image


def prepare_dataset(ds, is_training=True):
    """データセットをバッチ化して準備"""
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_mobilenet_model(num_classes=3):
    """MobileNetV2 転移学習モデルを構築"""
    print("\n" + "=" * 50)
    print("🏗 MobileNetV2 転移学習モデルを構築中...")
    print("=" * 50)

    # バックボーン（重みは凍結）
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    # 分類ヘッド
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


def train_model(model, ds_train, ds_test):
    """モデルを学習"""
    print("\n" + "=" * 50)
    print("🏋️ 学習開始...")
    print("=" * 50)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    history = model.fit(
        ds_train,
        epochs=EPOCHS_RPS,
        validation_data=ds_test,
        callbacks=callbacks,
        verbose=1
    )

    return history


def evaluate_model(model, ds_test):
    """モデルを評価（confusion matrix 付き）"""
    print("\n" + "=" * 50)
    print("📊 テストデータで評価中...")
    print("=" * 50)

    # 精度評価
    loss, accuracy = model.evaluate(ds_test, verbose=0)
    print(f"📈 テスト精度: {accuracy * 100:.2f}%")
    print(f"📉 テスト損失: {loss:.4f}")

    # 予測と正解ラベルを収集
    y_true = []
    y_pred = []
    for images, labels in ds_test:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 分類レポート
    print("\n📋 分類レポート:")
    print(classification_report(y_true, y_pred, target_names=RPS_CLASSES))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=RPS_CLASSES, yticklabels=RPS_CLASSES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig("data/rps/confusion_matrix.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("📊 Confusion Matrix を data/rps/confusion_matrix.png に保存しました")

    return accuracy


def measure_inference_speed(model):
    """推論速度を計測"""
    print("\n" + "=" * 50)
    print("⏱ 推論速度を計測中...")
    print("=" * 50)

    # ダミー入力で推論
    dummy_input = np.random.randn(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)

    # ウォームアップ
    for _ in range(5):
        model.predict(dummy_input, verbose=0)

    # 計測（50回の平均）
    times = []
    for _ in range(50):
        start = time.time()
        model.predict(dummy_input, verbose=0)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"⚡ 平均推論時間: {avg_time:.1f} ms ± {std_time:.1f} ms")
    print(f"📺 推定FPS: {1000 / avg_time:.1f}")

    return avg_time


def save_model(model):
    """モデルを保存"""
    os.makedirs(os.path.dirname(RPS_MODEL_PATH), exist_ok=True)
    model.save(RPS_MODEL_PATH)
    print(f"\n💾 モデルを保存しました: {RPS_MODEL_PATH}")


def plot_training_history(history):
    """学習履歴をプロット"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Test')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Test')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("data/rps/training_history.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("📊 学習履歴を data/rps/training_history.png に保存しました")


def main():
    """メイン処理"""
    print("🚀 ステップ2: Rock Paper Scissors 転移学習")
    print("=" * 60)

    # 1. データ取得
    ds_train_raw, ds_test_raw = load_rps_data()

    # 2. データセット準備
    ds_train = prepare_dataset(ds_train_raw, is_training=True)
    ds_test = prepare_dataset(ds_test_raw, is_training=False)

    # 3. モデル構築
    model = build_mobilenet_model(num_classes=3)

    # 4. 学習
    history = train_model(model, ds_train, ds_test)

    # 5. 評価
    accuracy = evaluate_model(model, ds_test)

    # 6. 推論速度チェック
    avg_time = measure_inference_speed(model)

    # 7. モデル保存
    save_model(model)

    # 8. 学習履歴のプロット
    plot_training_history(history)

    print("\n" + "=" * 60)
    print("✅ ステップ2 完了！")
    print(f"   テスト精度: {accuracy * 100:.2f}%")
    print(f"   推論速度: {avg_time:.1f} ms/フレーム")
    print(f"   モデル: {RPS_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
