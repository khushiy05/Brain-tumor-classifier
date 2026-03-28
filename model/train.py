"""
Brain Tumor Classification - CNN Model Training
================================================
Trains a CNN to classify brain MRI scans into 4 categories:
  - glioma
  - meningioma
  - pituitary
  - no_tumor

Dataset: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─── CONFIG ────────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 20
NUM_CLASSES = 4
CLASSES     = ["glioma", "meningioma", "notumor", "pituitary"]

TRAIN_DIR = "data/Training"
TEST_DIR  = "data/Testing"
MODEL_SAVE = "model/brain_tumor_cnn.h5"
# ───────────────────────────────────────────────────────────────────────────────


def build_model():
    """
    CNN Architecture:
    - 3 Convolutional blocks (Conv2D + BatchNorm + MaxPool)
    - Global Average Pooling
    - Dense classifier head
    - Dropout for regularization
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(*IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.4),

        # Classifier head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_data_generators():
    """Data augmentation for training, simple rescaling for validation."""
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', classes=CLASSES
    )
    val_data = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', classes=CLASSES
    )
    test_data = test_gen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', classes=CLASSES, shuffle=False
    )
    return train_data, val_data, test_data


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'],    label='Train Acc')
    axes[0].plot(history.history['val_accuracy'],label='Val Acc')
    axes[0].set_title('Accuracy'); axes[0].legend()

    axes[1].plot(history.history['loss'],    label='Train Loss')
    axes[1].plot(history.history['val_loss'],label='Val Loss')
    axes[1].set_title('Loss'); axes[1].legend()

    plt.tight_layout()
    plt.savefig('assets/training_history.png', dpi=150)
    plt.show()
    print("✅ Training history saved → assets/training_history.png")


def plot_confusion_matrix(model, test_data):
    y_pred  = np.argmax(model.predict(test_data), axis=1)
    y_true  = test_data.classes
    cm      = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('assets/confusion_matrix.png', dpi=150)
    plt.show()
    print("✅ Confusion matrix saved → assets/confusion_matrix.png")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASSES))


def main():
    print("🧠 Brain Tumor Classification - Training Started\n")
    os.makedirs("assets", exist_ok=True)
    os.makedirs("model",  exist_ok=True)

    train_data, val_data, test_data = get_data_generators()
    model = build_model()
    model.summary()

    cbs = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
        callbacks.ModelCheckpoint(MODEL_SAVE, save_best_only=True)
    ]

    history = model.fit(
        train_data, validation_data=val_data,
        epochs=EPOCHS, callbacks=cbs
    )

    plot_training_history(history)

    print("\n📊 Evaluating on test set...")
    loss, acc = model.evaluate(test_data)
    print(f"Test Accuracy: {acc*100:.2f}%  |  Test Loss: {loss:.4f}")

    plot_confusion_matrix(model, test_data)
    print(f"\n✅ Model saved → {MODEL_SAVE}")


if __name__ == "__main__":
    main()
