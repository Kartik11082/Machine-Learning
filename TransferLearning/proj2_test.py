import pandas as pd
import argparse
import tensorflow as tf
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Prevent crash on corrupted images


def load_model_weights(model_path, weights=None):
    model = tf.keras.models.load_model(model_path)
    model.summary()
    return model


def decode_img(img_path, img_height, img_width):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def get_images_labels(df, classes, img_height, img_width):
    class_list = sorted(list(classes))
    label_map = {label: idx for idx, label in enumerate(class_list)}

    image_tensors = []
    label_tensors = []

    for _, row in df.iterrows():
        img_path = row["image_path"]
        label = row["label"]

        if not os.path.exists(img_path):
            print(f"Warning: File not found: {img_path}")
            continue

        try:
            img = decode_img(img_path, img_height, img_width)
            one_hot = tf.keras.utils.to_categorical(
                label_map[label], num_classes=len(classes)
            )
            image_tensors.append(img)
            label_tensors.append(one_hot)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue

    if not image_tensors or not label_tensors:
        print("ERROR: No valid test images loaded!")
        return None, None

    test_images = tf.stack(image_tensors)
    test_labels = tf.convert_to_tensor(label_tensors)
    return test_images, test_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer Learning Test")
    parser.add_argument(
        "--model", type=str, default="my_model.h5", help="Saved model"
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="Weight file if needed"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="./sample_test_data/mushrooms_test.csv",
        help="CSV file with true labels",
    )
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_csv)
    classes = {
        "Agaricus",
        "Amanita",
        "Boletus",
        "Cortinarius",
        "Entoloma",
        "Hygrocybe",
        "Lactarius",
        "Russula",
        "Suillus",
    }

    IMG_HEIGHT, IMG_WIDTH = 224, 224
    print("Loading test images and labels...")
    test_images, test_labels = get_images_labels(
        test_df, classes, IMG_HEIGHT, IMG_WIDTH
    )

    if test_images is None or test_labels is None:
        print("Exiting due to missing or invalid test data.")
        exit(1)

    print("Loading model...")
    my_model = load_model_weights(args.model)

    print("Evaluating model...")
    loss, acc = my_model.evaluate(test_images, test_labels, verbose=2)
    print("Test model accuracy: {:5.2f}%".format(100 * acc))
