

# ==============================================================================# American Sign Language Recognition
# Dataset: Kaggle "American Sign Language Dataset" by Ayush Thakur
# Models:
# 1. CNN from scratch
# 2. MobileNetV2 Transfer Learning + Fine-tuning


# Install if needed:
# pip install tensorflow kagglehub scikit-learn matplotlib

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import kagglehub

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

print("TensorFlow:", tf.__version__)


# Download dataset
# Kaggle dataset slug associated with the ASL dataset.
# If you already downloaded it, replace DATASET_DIR with the folder
# containing the class subfolders.

path = kagglehub.dataset_download("ayuraj/asl-dataset")
print("Downloaded to:", path)

# Find a likely directory whose immediate subdirectories are class folders.
def find_class_directory(root):
    candidates = []
    for current, dirs, files in os.walk(root):
        image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        if len(dirs) >= 10:
            # Prefer directories whose child folders contain images.
            child_image_count = 0
            for d in dirs[:50]:
                dp = os.path.join(current, d)
                try:
                    child_image_count += sum(
                        f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                        for f in os.listdir(dp)
                    )
                except OSError:
                    pass
            if child_image_count > 0:
                candidates.append((len(dirs), child_image_count, current))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2]
    return root

DATASET_DIR = find_class_directory(path)
print("Using dataset directory:", DATASET_DIR)
print("Class folders:", sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
]))


# Dataset configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# First create 70% training and 30% temporary validation dataset.
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.30,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

temp_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.30,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

# Split the temporary 30% approximately equally into validation and test.
temp_batches = tf.data.experimental.cardinality(temp_ds).numpy()
split_batches = max(1, temp_batches // 2)
val_ds = temp_ds.take(split_batches)
test_ds = temp_ds.skip(split_batches)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("Number of classes:", NUM_CLASSES)
print("Classes:", class_names)


# Display sample images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    n = min(9, images.shape[0])
    for i in range(n):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
plt.tight_layout()
plt.show()


# Data augmentation
# Horizontal flipping is intentionally avoided because mirroring a hand sign
# can change its physical/semantic configuration.
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.05, 0.05),
], name="augmentation")


# MODEL 1: CNN FROM SCRATCH
cnn_model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(256, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation="softmax")
], name="cnn_from_scratch")

cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cnn_model.summary()


# Train CNN
CNN_EPOCHS = 20

cnn_history = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CNN_EPOCHS
)

cnn_loss, cnn_accuracy = cnn_model.evaluate(test_ds, verbose=1)
print(f"CNN Test Loss: {cnn_loss:.4f}")
print(f"CNN Test Accuracy: {cnn_accuracy * 100:.2f}%")


# CNN training curves
plt.figure(figsize=(8, 5))
plt.plot(cnn_history.history["accuracy"], label="Training Accuracy")
plt.plot(cnn_history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(cnn_history.history["loss"], label="Training Loss")
plt.plot(cnn_history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss")
plt.legend()
plt.grid()
plt.show()


# MODEL 2: MOBILENETV2 TRANSFER LEARNING
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

transfer_model = tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer")

transfer_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

transfer_model.summary()


# Train transfer-learning classifier
TL_EPOCHS = 10

transfer_history = transfer_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=TL_EPOCHS
)

tl_loss, tl_accuracy = transfer_model.evaluate(test_ds, verbose=1)
print(f"Transfer Learning Test Loss: {tl_loss:.4f}")
print(f"Transfer Learning Test Accuracy: {tl_accuracy * 100:.2f}%")


# Fine-tune the last 30 MobileNetV2 layers
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen for stable fine-tuning.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

transfer_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

FINE_TUNE_EPOCHS = 5

fine_history = transfer_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS
)

fine_loss, fine_accuracy = transfer_model.evaluate(test_ds, verbose=1)
print(f"Fine-tuned Test Loss: {fine_loss:.4f}")
print(f"Fine-tuned Test Accuracy: {fine_accuracy * 100:.2f}%")


# Compare models
model_names = ["CNN", "MobileNetV2", "Fine-tuned MobileNetV2"]
accuracies = [cnn_accuracy * 100, tl_accuracy * 100, fine_accuracy * 100]

plt.figure(figsize=(9, 5))
plt.bar(model_names, accuracies)
plt.ylabel("Test Accuracy (%)")
plt.title("ASL Recognition Model Comparison")
plt.ylim(0, 100)
for i, acc in enumerate(accuracies):
    plt.text(i, min(acc + 1, 99), f"{acc:.2f}%", ha="center")
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()


# Classification report and confusion matrix for fine-tuned model
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = transfer_model.predict(images, verbose=0)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(np.argmax(predictions, axis=1).tolist())

print(classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    zero_division=0
))

cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

plt.figure(figsize=(14, 12))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Fine-tuned MobileNetV2")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(NUM_CLASSES), class_names, rotation=90)
plt.yticks(np.arange(NUM_CLASSES), class_names)
plt.colorbar()
plt.tight_layout()
plt.show()


# Predict one image
def predict_asl(image_path, model=transfer_model):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = tf.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")
    plt.show()

    print("Predicted Sign:", predicted_class)
    print(f"Confidence: {confidence:.2f}%")
    return predicted_class, confidence

# Example:
# predict_asl("sample_asl_image.jpg")


# Save models and class labels
cnn_model.save("asl_cnn_from_scratch.keras")
transfer_model.save("asl_mobilenetv2_transfer_learning.keras")

with open("asl_class_names.txt", "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print("Saved:")
print(" - asl_cnn_from_scratch.keras")
print(" - asl_mobilenetv2_transfer_learning.keras")
print(" - asl_class_names.txt")
