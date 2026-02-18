"""
ステップ1: MNIST 学習パイプライン（パイプラインの型作り）
目的: 学習→評価→保存→推論の一連の流れが正しく動くことを確認する。
※ Sign Language MNISTの代わりにKeras内蔵のFashion MNISTを使用
  （28×28グレースケール・多クラス分類として同じ構造）
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- 設定の読み込み ---
from config import MNIST_IMG_SIZE, BATCH_SIZE, EPOCHS_MNIST, MNIST_MODEL_PATH

# Fashion MNIST のクラス名（日本語）
CLASS_NAMES = [
    "Tシャツ", "ズボン", "プルオーバー", "ドレス", "コート",
    "サンダル", "シャツ", "スニーカー", "バッグ", "ブーツ"
]


def load_data():
    """Fashion MNIST データセットを読み込み"""
    print("=" * 50)
    print("📥 Fashion MNIST データセットを読み込み中...")
    print("=" * 50)

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # 正規化 & チャネル次元追加
    x_train = x_train.reshape(-1, MNIST_IMG_SIZE, MNIST_IMG_SIZE, 1).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, MNIST_IMG_SIZE, MNIST_IMG_SIZE, 1).astype(np.float32) / 255.0

    print(f"✅ 学習データ: {x_train.shape}, ラベル: {y_train.shape}")
    print(f"✅ テストデータ: {x_test.shape}, ラベル: {y_test.shape}")
    print(f"📊 クラス数: {len(np.unique(y_train))}")

    return (x_train, y_train), (x_test, y_test)


def build_cnn_model(num_classes):
    """小さめCNNモデルを構築"""
    model = keras.Sequential([
        layers.Input(shape=(MNIST_IMG_SIZE, MNIST_IMG_SIZE, 1)),

        # ブロック1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ブロック2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # 全結合層
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(model, x_train, y_train, x_test, y_test):
    """モデルを学習"""
    print("\n" + "=" * 50)
    print("🏋️ 学習開始...")
    print("=" * 50)

    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_MNIST,
        validation_data=(x_test, y_test),
        verbose=1
    )

    return history


def evaluate_model(model, x_test, y_test):
    """モデルを評価"""
    print("\n" + "=" * 50)
    print("📊 テストデータで評価中...")
    print("=" * 50)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"📈 テスト精度: {accuracy * 100:.2f}%")
    print(f"📉 テスト損失: {loss:.4f}")

    return accuracy


def save_model(model):
    """モデルを保存"""
    os.makedirs(os.path.dirname(MNIST_MODEL_PATH), exist_ok=True)
    model.save(MNIST_MODEL_PATH)
    print(f"\n💾 モデルを保存しました: {MNIST_MODEL_PATH}")


def test_single_prediction(model, x_test, y_test):
    """1枚推論テスト"""
    print("\n" + "=" * 50)
    print("🔍 1枚推論テスト")
    print("=" * 50)

    idx = np.random.randint(0, len(x_test))
    img = x_test[idx:idx+1]
    true_label = y_test[idx]

    probs = model.predict(img, verbose=0)
    pred_label = np.argmax(probs[0])
    confidence = np.max(probs[0])

    print(f"📌 正解: {CLASS_NAMES[true_label]} (ラベル {true_label})")
    print(f"🤖 予測: {CLASS_NAMES[pred_label]} (ラベル {pred_label})")
    print(f"💯 確信度: {confidence * 100:.2f}%")
    print(f"{'✅ 正解!' if pred_label == true_label else '❌ 不正解'}")

    # 画像を保存
    plt.figure(figsize=(4, 4))
    plt.imshow(x_test[idx].reshape(MNIST_IMG_SIZE, MNIST_IMG_SIZE), cmap='gray')
    plt.title(f"正解: {CLASS_NAMES[true_label]} / 予測: {CLASS_NAMES[pred_label]} ({confidence*100:.1f}%)")
    plt.axis('off')
    plt.savefig("data/mnist/sample_prediction.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("📸 推論結果を data/mnist/sample_prediction.png に保存しました")


def plot_training_history(history):
    """学習履歴をプロット"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='学習データ')
    ax1.plot(history.history['val_accuracy'], label='テストデータ')
    ax1.set_title('精度（Accuracy）')
    ax1.set_xlabel('エポック')
    ax1.set_ylabel('精度')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='学習データ')
    ax2.plot(history.history['val_loss'], label='テストデータ')
    ax2.set_title('損失（Loss）')
    ax2.set_xlabel('エポック')
    ax2.set_ylabel('損失')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("data/mnist/training_history.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("📊 学習履歴を data/mnist/training_history.png に保存しました")


def main():
    """メイン処理"""
    print("🚀 ステップ1: MNIST 学習パイプライン（型作り）")
    print("=" * 60)

    # 1. データ取得
    (x_train, y_train), (x_test, y_test) = load_data()

    # 2. モデル構築
    num_classes = len(np.unique(y_train))
    model = build_cnn_model(num_classes)
    model.summary()

    # 3. 学習
    history = train_model(model, x_train, y_train, x_test, y_test)

    # 4. 評価
    accuracy = evaluate_model(model, x_test, y_test)

    # 5. モデル保存
    save_model(model)

    # 6. 学習履歴のプロット
    plot_training_history(history)

    # 7. 1枚推論テスト
    test_single_prediction(model, x_test, y_test)

    print("\n" + "=" * 60)
    print("✅ ステップ1 完了！")
    print(f"   テスト精度: {accuracy * 100:.2f}%")
    print(f"   モデル: {MNIST_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
