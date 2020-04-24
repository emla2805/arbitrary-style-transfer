import tensorflow as tf

from PIL import Image
from argparse import ArgumentParser

from networks import TransferNet
from utils import load_img

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log-dir", default="model")
    parser.add_argument("--content-image", required=True)
    parser.add_argument("--style-image", required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--alpha", default=1.0, type=float)
    args = parser.parse_args()

    content_image = load_img(args.content_image)
    style_image = load_img(args.style_image)

    content_layer = "block4_conv1"
    transformer = TransferNet(content_layer)
    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt.restore(tf.train.latest_checkpoint(args.log_dir)).expect_partial()

    stylized_image = transformer(content_image, style_image, alpha=args.alpha)
    stylized_image = tf.cast(
        tf.squeeze(stylized_image), tf.uint8
    ).numpy()

    img = Image.fromarray(stylized_image, mode="RGB")
    img.save(args.output_image)
