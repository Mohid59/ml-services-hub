"""Standalone training script for the gender classifier.

Two architectures (pick with --arch):
  scratch    - small Conv2D/MaxPool CNN trained from zero (fast baseline)
  mobilenet  - MobileNetV2 transfer learning: frozen ImageNet base + new head,
               then fine-tune the top of the base at a low learning rate.

Expected dataset layout (see services/prepare_gender_data.py):

    data/
      train/{female,male}/*.jpg
      val/{female,male}/*.jpg

Run:
    python services/train_gender_cnn.py --arch mobilenet --img-size 96 --epochs 20

Output: models/gender_cnn.h5 (auto-loaded by the web service; input size is
read from the saved model, so any --img-size works without config changes).
"""
import argparse
import os
import sys

# Make project root importable when run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config  # noqa: E402


def build_scratch(input_shape):
    """Small Conv2D/MaxPool CNN -> Dense -> sigmoid. P(male) output."""
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
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def build_mobilenet(input_shape):
    """MobileNetV2 (ImageNet) base, frozen, + small classification head.

    Returns (model, base) so the caller can unfreeze for fine-tuning.
    Note: inputs stay in [0,1]; a Rescaling layer maps to MobileNet's
    expected [-1,1] so the service preprocessing stays identical.
    """
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(include_top=False, weights="imagenet",
                       input_shape=input_shape)
    base.trainable = False
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(2.0, offset=-1.0),   # [0,1] -> [-1,1]
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model, base


def main():
    parser = argparse.ArgumentParser(description="Train gender classifier")
    parser.add_argument("--data", default="data", help="dataset root dir")
    parser.add_argument("--arch", choices=["scratch", "mobilenet"],
                        default="scratch")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=6,
                        help="extra low-LR epochs with top of base unfrozen "
                             "(mobilenet only)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", default=config.GENDER_CNN_PATH)
    args = parser.parse_args()

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    img_size = (args.img_size, args.img_size)
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

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=4,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                          min_lr=1e-6),
    ]

    input_shape = (*img_size, 3)
    if args.arch == "scratch":
        model = build_scratch(input_shape)
        model.summary()
        model.fit(train_flow, validation_data=val_flow, epochs=args.epochs,
                  callbacks=callbacks)
    else:
        model, base = build_mobilenet(input_shape)
        model.summary()
        # Phase 1: train the new head with the base frozen.
        model.fit(train_flow, validation_data=val_flow, epochs=args.epochs,
                  callbacks=callbacks)
        # Phase 2: unfreeze the top ~30% of the base, fine-tune at low LR.
        if args.fine_tune_epochs > 0:
            base.trainable = True
            cutoff = int(len(base.layers) * 0.7)
            for layer in base.layers[:cutoff]:
                layer.trainable = False
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                          loss="binary_crossentropy", metrics=["accuracy"])
            print(f"[fine-tune] unfroze {len(base.layers) - cutoff} layers")
            model.fit(train_flow, validation_data=val_flow,
                      epochs=args.fine_tune_epochs, callbacks=callbacks)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    print(f"Saved trained model -> {args.out}")


if __name__ == "__main__":
    main()
