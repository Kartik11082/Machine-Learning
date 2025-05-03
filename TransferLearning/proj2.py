from google.colab import drive

drive.mount("/content/drive")

import os

import tensorflow as tf
from PIL import Image, ImageFile
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

ImageFile.LOAD_TRUNCATED_IMAGES = True
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))


def remove_corrupted_images(base_dir):
    count = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    filepath = os.path.join(root, file)
                    img = Image.open(filepath)
                    img.verify()  # This will raise an exception for corrupt files
                except Exception as e:
                    print(f"Removing corrupt image: {filepath} ({e})")
                    os.remove(filepath)
                    count += 1
    print(f"Removed {count} corrupted image(s)")


# Run this once before training to clean dataset
remove_corrupted_images("Mushrooms")

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     validation_split=0.2,
#     rotation_range=20,
#     zoom_range=0.2,
#     horizontal_flip=True,
# )

# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     validation_split=0.2,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.3,
#     horizontal_flip=True,
#     fill_mode="nearest",
# )

# Image augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.4,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)

# Create training and validation generators from dataset
train_generator = train_datagen.flow_from_directory(
    "/content/drive/MyDrive/Colab Notebooks/Mushrooms",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
)

val_generator = train_datagen.flow_from_directory(
    "/content/drive/MyDrive/Colab Notebooks/Mushrooms",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet"
)


# Fine-tuning configuration: freeze early layers, unfreeze last 100
unfreezed_layers = -100
for layer in base_model.layers[:unfreezed_layers]:  # freeze early 90% layers
    layer.trainable = False

for layer in base_model.layers[unfreezed_layers:]:  # unfreeze deeper high-level layers
    layer.trainable = True

# base_model = tf.keras.applications.EfficientNetB0(
#     input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet"
# )

# Freeze base
base_model.trainable = True


# model = models.Sequential(
#     [
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(128, activation="relu"),
#         layers.Dropout(0.5),
#         layers.Dense(9, activation="softmax"),  # 9 classes
#     ]
# )

# model = models.Sequential(
#     [
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(256, activation="relu"),
#         layers.BatchNormalization(),
#         layers.Dropout(0.3),
#         layers.Dense(128, activation="relu"),
#         layers.Dropout(0.3),
#         layers.Dense(9, activation="softmax"),  # 9 classes
#     ]
# )

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(9, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_generator, validation_data=val_generator, epochs=30)

base_model.trainable = True

optimizer = Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower LR
#     loss="categorical_crossentropy",
#     metrics=["accuracy"],
# )

# lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-7
# )

lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[lr_scheduler],
)


# Function to generate a unique filename for saved model
def get_incremental_filename(base_name, extension):
    counter = 1
    while True:
        filename = f"{base_name}_{counter}.{extension}"
        if not os.path.exists(filename):
            return filename
        counter += 1


# Usage
filename = get_incremental_filename("my_model", "h5")
model.save(filename)

model.save(f"/content/drive/MyDrive/Colab Notebooks/{filename}")
print(f"Model saved as {filename}")
