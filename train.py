import os
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_datasets as tfds

from networks import TransformerNet, VGG, Encoder, decoder
from utils import load_img, style_loss, content_loss

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
    vgg = VGG(content_layer, style_layers)
    transformer = TransformerNet(content_layer)

    vgg(test_style_images)

    def process_content(features):
        img = features["image"]
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, size=(512, 512))
        img = tf.image.random_crop(
            img, size=(args.image_size, args.image_size, 3)
        )
        return img

    def process_style(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size=(512, 512), preserve_aspect_ratio=True)
        img = tf.image.random_crop(
            img, size=(args.image_size, args.image_size, 3)
        )
        img = tf.image.random_flip_left_right(img)
        return img

    # Warning: Downloads the full coco/2014 dataset
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
    ckpt = tf.train.Checkpoint(optimizer=optimizer, transformer=transformer)
    manager = tf.train.CheckpointManager(ckpt, args.log_dir, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")

    summary_writer = tf.summary.create_file_writer(args.log_dir)

    with summary_writer.as_default():
        tf.summary.image(
            "content", test_content_images / 255.0, step=0, max_outputs=6
        )
        tf.summary.image(
            "style", test_style_images / 255.0, step=0, max_outputs=6
        )

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_style_loss = tf.keras.metrics.Mean(name="train_style_loss")
    train_content_loss = tf.keras.metrics.Mean(name="train_content_loss")

    @tf.function
    def train_step(content_img, style_img):
        t = transformer.encode(content_img, style_img, alpha=1.0)

        with tf.GradientTape() as tape:
            stylized_img = transformer.decode(t)

            _, style_feat_style = vgg(style_img)
            content_feat_stylized, style_feat_stylized = vgg(stylized_img)

            tot_style_loss = args.style_weight * style_loss(
                style_feat_style, style_feat_stylized
            )
            tot_content_loss = args.content_weight * content_loss(
                t, content_feat_stylized
            )
            loss = tot_style_loss + tot_content_loss

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables)
        )

        train_loss(loss)
        train_style_loss(tot_style_loss)
        train_content_loss(tot_content_loss)

    for step, (content_images, style_images) in enumerate(ds):
        new_lr = args.lr / (1.0 + args.lr_decay * step)
        optimizer.learning_rate.assign(new_lr)

        train_step(content_images, style_images)

        if step % args.log_freq == 0:
            with summary_writer.as_default():
                tf.summary.scalar("loss/total", train_loss.result(), step=step)
                tf.summary.scalar(
                    "loss/style", train_style_loss.result(), step=step
                )
                tf.summary.scalar(
                    "loss/content", train_content_loss.result(), step=step
                )
                stylized_images = transformer(
                    test_content_images, test_style_images
                )
                tf.summary.image(
                    "stylized",
                    stylized_images / 255.0,
                    step=step,
                    max_outputs=6,
                )

            print(
                f"Step {step}, "
                f"Loss: {train_loss.result()}, "
                f"Style Loss: {train_style_loss.result()}, "
                f"Content Loss: {train_content_loss.result()}"
            )
            print(f"Saved checkpoint: {manager.save()}")

            train_loss.reset_states()
            train_style_loss.reset_states()
            train_content_loss.reset_states()
