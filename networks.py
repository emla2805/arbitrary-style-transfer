import tensorflow as tf
from tensorflow.keras.applications import vgg19, VGG19
from tensorflow.keras.layers import Conv2D, UpSampling2D


class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=1, **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = padding

    def compute_output_shape(self, s):
        return s[0], s[1] + 2 * self.padding, s[2] + 2 * self.padding, s[3]

    def call(self, x):
        return tf.pad(
            x,
            [
                [0, 0],
                [self.padding, self.padding],
                [self.padding, self.padding],
                [0, 0],
            ],
            "REFLECT",
        )


class TransformerNet(tf.keras.Model):
    def __init__(self, style_layers, content_layers):
        super(TransformerNet, self).__init__()
        self.encoder = Encoder(style_layers, content_layers)
        self.decoder = decoder()

    def norm(self, content_feat, style_feat, alpha):
        t = adaptive_instance_normalization(content_feat, style_feat)
        t = alpha * t + (1 - alpha) * content_feat
        return t

    def call(self, content_image, style_image, alpha=1.0):
        _, content_feat = self.encoder(content_image)
        style_feat, _ = self.encoder(style_image)

        t = self.norm(content_feat[0], style_feat[-1], alpha)
        g_t = self.decoder(t)
        return g_t


def adaptive_instance_normalization(content_feat, style_feat, epsilon=1e-5):
    content_mean, content_variance = tf.nn.moments(
        content_feat, axes=[1, 2], keepdims=True
    )
    style_mean, style_variance = tf.nn.moments(
        style_feat, axes=[1, 2], keepdims=True
    )
    style_std = tf.math.sqrt(style_variance)
    normalized = (content_feat - content_mean) * tf.math.rsqrt(
        content_variance + epsilon
    )
    return normalized * style_std + style_mean


def decoder():
    return tf.keras.Sequential(
        [
            ReflectionPadding2D(),
            Conv2D(256, (3, 3), activation="relu"),
            UpSampling2D(size=2),
            ReflectionPadding2D(),
            Conv2D(256, (3, 3), activation="relu"),
            ReflectionPadding2D(),
            Conv2D(256, (3, 3), activation="relu"),
            ReflectionPadding2D(),
            Conv2D(256, (3, 3), activation="relu"),
            ReflectionPadding2D(),
            Conv2D(128, (3, 3), activation="relu"),
            UpSampling2D(size=2),
            ReflectionPadding2D(),
            Conv2D(128, (3, 3), activation="relu"),
            ReflectionPadding2D(),
            Conv2D(64, (3, 3), activation="relu"),
            UpSampling2D(size=2),
            ReflectionPadding2D(),
            Conv2D(64, (3, 3), activation="relu"),
            ReflectionPadding2D(),
            Conv2D(3, (3, 3)),
        ]
    )


class Encoder(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(Encoder, self).__init__()
        vgg = VGG19(include_top=False, weights="imagenet")

        style_outputs = [vgg.get_layer(name).output for name in style_layers]
        content_outputs = [
            vgg.get_layer(name).output for name in content_layers
        ]

        self.vgg = tf.keras.Model(
            [vgg.input], [style_outputs, content_outputs]
        )
        self.vgg.trainable = False

    def call(self, inputs):
        preprocessed_input = vgg19.preprocess_input(inputs)
        style_outputs, content_outputs = self.vgg(preprocessed_input)
        return style_outputs, content_outputs
