import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
PBN_PATH = "https://github.com/zo7/painter-by-numbers/releases/download/data-v1.0/{split}.tgz"


def load_pbn(split="train"):
    data_dir = tf.keras.utils.get_file(
        origin=PBN_PATH.format(split=split), fname=split, untar=True
    )
    ds_list = tf.data.Dataset.list_files(f"{data_dir}/*.jpg")
    ds = ds_list.map(process_path, num_parallel_calls=AUTOTUNE)
    return ds


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    return img


def load_img(file_path):
    img = process_path(file_path)
    img = img[tf.newaxis, :]
    return img


def mean_std_loss(feat, feat_stylized):
    feat_mean, feat_variance = tf.nn.moments(feat, axes=[1, 2], keepdims=True)
    feat_stylized_mean, feat_stylized_variance = tf.nn.moments(
        feat_stylized, axes=[1, 2], keepdims=True
    )
    feat_std = tf.math.sqrt(feat_variance)
    feat_stylized_std = tf.math.sqrt(feat_stylized_variance)
    loss = tf.losses.mse(feat_stylized_mean, feat_mean) + tf.losses.mse(
        feat_stylized_std, feat_std
    )
    return loss


def style_loss(feat, feat_stylized):
    return tf.add_n(
        [
            mean_std_loss(f, f_stylized)
            for f, f_stylized in zip(feat, feat_stylized)
        ]
    )


def content_loss(feat, feat_stylized):
    return tf.add_n(
        [
            tf.losses.mse(f, f_stylized)
            for f, f_stylized in zip(feat, feat_stylized)
        ]
    )
