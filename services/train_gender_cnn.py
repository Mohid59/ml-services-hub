"""Standalone training script for the gender CNN.

Expected dataset layout (binary classification, folder-per-class):

    data/
      train/
        female/  *.jpg
        male/    *.jpg
      val/
        female/  *.jpg
        male/    *.jpg

Where to get data: any face/gender dataset works, e.g. Kaggle
"Gender Classification Dataset" or a CelebA subset split into male/female.
Place images into the folders above (class names must be 'female' and 'male'
so the label order matches the inference service).

Run:
    python services/train_gender_cnn.py --data data --epochs 10

Output: models/gender_cnn.h5  (auto-loaded by the web service on next request)
"""
import argparse
import os
import sys

# Make project root importable when run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config  # noqa: E402


def build_cnn(input_shape):
    """A small Conv2D/MaxPool CNN -> Dense -> sigmoid for binary output."""
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),  # P(male)
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train gender CNN")
    parser.add_argument("--data", default="data", help="dataset root dir")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    img_size = config.GENDER_IMG_SIZE
    train_dir = os.path.join(args.data, "train")
    val_dir = os.path.join(args.data, "val")
    for d in (train_dir, val_dir):
        if not os.path.isdir(d):
            sys.exit(f"Missing directory: {d} (see this script's docstring).")

    # Augmentation on train, just rescale on val.
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255, rotation_range=15, width_shift_range=0.1,
        height_shift_range=0.1, horizontal_flip=True, zoom_range=0.1)
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    # classes=['female','male'] pins label order to match the service.
    train_flow = train_gen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=args.batch,
        class_mode="binary", classes=["female", "male"])
    val_flow = val_gen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=args.batch,
        class_mode="binary", classes=["female", "male"])

    model = build_cnn((*img_size, 3))
    model.summary()
    model.fit(train_flow, validation_data=val_flow, epochs=args.epochs)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model.save(config.GENDER_CNN_PATH)
    print(f"Saved trained model -> {config.GENDER_CNN_PATH}")


if __name__ == "__main__":
    main()
