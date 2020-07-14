import os
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import TensorBoard

from networks import TransferNet, VGG, Encoder, decoder
from utils import load_img, resize, style_loss, content_loss

AUTOTUNE = tf.data.experimental.AUTOTUNE


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log-dir", default="model")
    parser.add_argument("--style-dir", required=True)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr-decay", default=5e-5, type=float)
    parser.add_argument("--max-steps", default=160_000, type=int)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--content-weight", default=1, type=float)
    parser.add_argument("--style-weight", default=10, type=float)
    parser.add_argument("--log-freq", default=500, type=int)
    args = parser.parse_args()

    content_paths = ["avril_cropped.jpg", "chicago_cropped.jpg"]
    style_paths = ["impronte_d_artista_cropped.jpg", "ashville_cropped.jpg"]
    test_content_images = tf.concat(
        [load_img(f"images/content/{f}") for f in content_paths], axis=0
    )
    test_style_images = tf.concat(
        [load_img(f"images/style/{f}") for f in style_paths], axis=0
    )

    content_layer = "block4_conv1"  # relu-4-1
    style_layers = [
        "block1_conv1",  # relu1-1
        "block2_conv1",  # relu2-1
        "block3_conv1",  # relu3-1
        "block4_conv1",  # relu4-1
    ]
    encoder = Encoder(content_layer)
    decoder = decoder()
    vgg = VGG(content_layer, style_layers)
    transformer = TransferNet(
        encoder,
        decoder,
        vgg,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
    )

    def resize_and_crop(img, min_size):
        img = resize(img, min_size=min_size)
        img = tf.image.random_crop(
            img, size=(args.image_size, args.image_size, 3)
        )
        img = tf.cast(img, tf.float32)
        return img

    def process_content(features):
        img = features["image"]
        img = resize_and_crop(img, min_size=286)
        return img

    def process_style(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = resize_and_crop(img, min_size=512)
        return img

    ds_coco = (
        tfds.load("coco/2014", split="train")
        .map(process_content, num_parallel_calls=AUTOTUNE)
        .repeat()
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )

    ds_pbn = (
        tf.data.Dataset.list_files(os.path.join(args.style_dir, "*.jpg"))
        .map(process_style, num_parallel_calls=AUTOTUNE)
        # Ignore too large or corrupt image files
        .apply(tf.data.experimental.ignore_errors())
        .repeat()
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )

    ds = tf.data.Dataset.zip((ds_coco, ds_pbn))

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    transformer.compile(
        optimizer=optimizer,
        style_loss_fn=style_loss,
        content_loss_fn=content_loss,
    )

    class TransferMonitor(tf.keras.callbacks.Callback):
        def __init__(self, log_dir, content_images, style_images):
            super().__init__()
            self.file_writer = tf.summary.create_file_writer(log_dir)
            self.images = [content_images, style_images]

        def on_train_batch_end(self, batch, logs=None):
            stylized_images = self.model(self.images)

            with self.file_writer.as_default():
                tf.summary.image(
                    "stylized",
                    stylized_images / 255.0,
                    max_outputs=6,
                    step=batch,
                )

    transformer.fit(
        ds,
        steps_per_epoch=args.max_steps,
        callbacks=[
            TensorBoard(
                log_dir=args.log_dir,
                update_freq=args.log_freq,
                profile_batch=0,
            ),
            TransferMonitor(
                args.log_dir, test_content_images, test_style_images
            ),
        ],
    )
