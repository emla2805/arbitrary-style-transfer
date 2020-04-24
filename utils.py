import tensorflow as tf


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    return img


def load_img(file_path):
    img = process_path(file_path)
    img = img[tf.newaxis, :]
    return img


def resize(img, min_size=512):
    """ Resize image and keep aspect ratio """
    width, height, _ = tf.unstack(tf.shape(img), num=3)
    if height < width:
        new_height = min_size
        new_width = int(width * new_height / height)
    else:
        new_width = min_size
        new_height = int(height * new_width / width)

    img = tf.image.resize(img, size=(new_width, new_height))
    return img


def mean_std_loss(feat, feat_stylized, epsilon=1e-5):
    feat_mean, feat_variance = tf.nn.moments(feat, axes=[1, 2])
    feat_stylized_mean, feat_stylized_variance = tf.nn.moments(
        feat_stylized, axes=[1, 2]
    )
    feat_std = tf.math.sqrt(feat_variance + epsilon)
    feat_stylized_std = tf.math.sqrt(feat_stylized_variance + epsilon)

    loss = tf.losses.mse(feat_stylized_mean, feat_mean) + tf.losses.mse(
        feat_stylized_std, feat_std
    )
    return loss


def style_loss(feat, feat_stylized):
    return tf.reduce_sum(
        [
            mean_std_loss(f, f_stylized)
            for f, f_stylized in zip(feat, feat_stylized)
        ]
    )


def content_loss(feat, feat_stylized):
    return tf.reduce_mean(tf.square(feat - feat_stylized), axis=[1, 2, 3])
