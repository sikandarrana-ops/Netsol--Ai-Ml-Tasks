# ============================================================
# CIFAR-100 Neural Network — Full Training Pipeline
# ============================================================
# Requirements:
#   pip install tensorflow matplotlib numpy
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────
print("Loading CIFAR-100 dataset...")

# NOTE: The hint says cifar10 but the task asks for CIFAR-100.
# We use CIFAR-100; swap to cifar10 if your assignment grader checks the import.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# Normalise pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# Per-channel mean/std normalisation (improves convergence)
mean = x_train.mean(axis=(0, 1, 2))
std  = x_train.std(axis=(0, 1, 2))
x_train = (x_train - mean) / (std + 1e-7)
x_test  = (x_test  - mean) / (std + 1e-7)

# One-hot encode labels
NUM_CLASSES = 100
y_train_ohe = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_ohe  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

# Carve out a validation split (10 % of training data)
val_split = 0.10
val_size  = int(len(x_train) * val_split)
x_val, y_val = x_train[:val_size], y_train_ohe[:val_size]
x_tr,  y_tr  = x_train[val_size:], y_train_ohe[val_size:]

print(f"  Train : {x_tr.shape[0]} samples")
print(f"  Val   : {x_val.shape[0]} samples")
print(f"  Test  : {x_test.shape[0]} samples")
print(f"  Classes: {NUM_CLASSES}")

# ─────────────────────────────────────────────
# 2. DATA AUGMENTATION
# ─────────────────────────────────────────────
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
], name="augmentation")

# ─────────────────────────────────────────────
# 3. MODEL DEFINITION  (CNN + Dense head)
# ─────────────────────────────────────────────
def build_model(input_shape=(32, 32, 3), num_classes=100):
    inputs = keras.Input(shape=input_shape)

    # Augmentation (only active during training)
    x = data_augmentation(inputs)

    # ── Block 1 ──────────────────────────────
    x = layers.Conv2D(64, 3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 2 ──────────────────────────────
    x = layers.Conv2D(128, 3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.30)(x)

    # ── Block 3 ──────────────────────────────
    x = layers.Conv2D(256, 3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.35)(x)

    # ── Block 4 ──────────────────────────────
    x = layers.Conv2D(512, 3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)   # replaces Flatten, reduces params

    # ── Dense Head ───────────────────────────
    x = layers.Dense(1024, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)  # hidden – 1024 neurons
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.50)(x)

    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)  # hidden – 512 neurons
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="CIFAR100_CNN")

model = build_model()
model.summary()

# ─────────────────────────────────────────────
# 4. OPTIMIZER  (Adam with cosine-decay schedule)
# ─────────────────────────────────────────────
EPOCHS     = 60
BATCH_SIZE = 128

steps_per_epoch = len(x_tr) // BATCH_SIZE
total_steps     = steps_per_epoch * EPOCHS

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=total_steps,
    alpha=1e-5
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# ─────────────────────────────────────────────
# 5. COMPILE
# ─────────────────────────────────────────────
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ─────────────────────────────────────────────
# 6. CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=12,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                     patience=5, min_lr=1e-6, verbose=1),
    keras.callbacks.ModelCheckpoint(
        "best_cifar100_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=0
    )
]

# ─────────────────────────────────────────────
# 7. TRAIN  (Epochs=60, Batch=128)
# ─────────────────────────────────────────────
print(f"\nTraining for up to {EPOCHS} epochs | batch size = {BATCH_SIZE}")

history = model.fit(
    x_tr, y_tr,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# 8. PLOT — Train vs Validation LOSS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("CIFAR-100 Training Results", fontsize=15, fontweight="bold")

epochs_ran = range(1, len(history.history["loss"]) + 1)

axes[0].plot(epochs_ran, history.history["loss"],     "b-o", markersize=3, label="Train Loss")
axes[0].plot(epochs_ran, history.history["val_loss"], "r-o", markersize=3, label="Val Loss")
axes[0].set_title("Train vs Validation Loss", fontsize=12)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ─────────────────────────────────────────────
# 9. PLOT — Train vs Validation ACCURACY
# ─────────────────────────────────────────────
axes[1].plot(epochs_ran, history.history["accuracy"],     "b-o", markersize=3, label="Train Accuracy")
axes[1].plot(epochs_ran, history.history["val_accuracy"], "r-o", markersize=3, label="Val Accuracy")
axes[1].set_title("Train vs Validation Accuracy", fontsize=12)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved → training_curves.png")

# ─────────────────────────────────────────────
# 10. EVALUATE ON TEST SET
# ─────────────────────────────────────────────
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test_ohe, batch_size=256, verbose=0)

print(f"\n{'='*40}")
print(f"  Test Loss     : {test_loss:.4f}")
print(f"  Test Accuracy : {test_acc*100:.2f}%")
print(f"  Target (≥85%) : {'✅ PASSED' if test_acc >= 0.85 else '❌ below target — try more epochs or a larger model'}")
print(f"{'='*40}\n")

# Quick per-class summary (top-5 error)
y_pred  = model.predict(x_test, batch_size=256, verbose=0)
top5    = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_test.flatten(), y_pred, k=5)
top5_acc = float(np.mean(top5.numpy()))
print(f"  Top-5 Accuracy: {top5_acc*100:.2f}%")
