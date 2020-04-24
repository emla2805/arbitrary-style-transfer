import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from argparse import ArgumentParser

from networks import TransferNet
from utils import load_img

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log-dir", default="model")
    parser.add_argument("--content-image", required=True)
    parser.add_argument("--style-image", required=True)
    args = parser.parse_args()

    content_image = load_img(args.content_image)
    style_image = load_img(args.style_image)

    content_layer = "block4_conv1"

    transformer = TransferNet(content_layer)
    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt.restore(tf.train.latest_checkpoint(args.log_dir)).expect_partial()

    images = []
    for i in range(0, 11):
        alpha = i / 10
        t = transformer.encode(content_image, style_image, alpha=alpha)
        stylized_image = transformer.decode(t)
        stylized_image = tf.cast(tf.squeeze(stylized_image), tf.uint8).numpy()

        img = Image.fromarray(stylized_image, mode="RGB")

        w, h = img.size

        canvas = Image.new("RGB", (w, h + 50), "white")
        canvas.paste(img)

        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(
            "/Users/emil.larsson/Library/Fonts/FiraCode-Medium.ttf",
            size=18,
            encoding="unic",
        )
        msg = f"alpha={alpha}"
        tw, th = font.getsize(msg)
        draw.text(xy=((w - tw)/2, h + 15), text=msg, fill="black", font=font)
        images.append(canvas)

    images_tot = images + images[::-1]

    images_tot[0].save(
        "test.gif",
        save_all=True,
        append_images=images_tot[1:],
        optimize=False,
        duration=300,
        loop=0,
    )
