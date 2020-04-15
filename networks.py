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
    def __init__(self, content_layer):
        super(TransformerNet, self).__init__()
        self.encoder = Encoder(content_layer)
        self.decoder = decoder()

    def encode(self, content_image, style_image, alpha):
        content_feat = self.encoder(content_image)
        style_feat = self.encoder(style_image)

        t = adaptive_instance_normalization(content_feat, style_feat)
        t = alpha * t + (1 - alpha) * content_feat
        return t

    def decode(self, t):
        return self.decoder(t)

    def call(self, content_image, style_image, alpha=1.0):
        t = self.encode(content_image, style_image, alpha)
        g_t = self.decode(t)
        return g_t


def adaptive_instance_normalization(content_feat, style_feat, epsilon=1e-5):
    content_mean, content_variance = tf.nn.moments(
        content_feat, axes=[1, 2], keepdims=True
    )
    style_mean, style_variance = tf.nn.moments(
        style_feat, axes=[1, 2], keepdims=True
    )
    style_std = tf.math.sqrt(style_variance)

    norm_content_feat = tf.nn.batch_normalization(
        content_feat,
        mean=content_mean,
        variance=content_variance,
        offset=style_mean,
        scale=style_std,  # Maybe std?
        variance_epsilon=epsilon,
    )
    return norm_content_feat


class Encoder(tf.keras.models.Model):
    def __init__(self, content_layer):
        super(Encoder, self).__init__()
        vgg = VGG19(include_top=False, weights="imagenet")

        self.vgg = tf.keras.Model(
            [vgg.input], [vgg.get_layer(content_layer).output]
        )
        self.vgg.trainable = False

    def call(self, inputs):
        preprocessed_input = vgg19.preprocess_input(inputs)
        return self.vgg(preprocessed_input)


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


class VGG(tf.keras.models.Model):
    def __init__(self, content_layer, style_layers):
        super(VGG, self).__init__()
        vgg = VGG19(include_top=False, weights="imagenet")

        content_outputs = [vgg.get_layer(content_layer).output]
        style_outputs = [vgg.get_layer(name).output for name in style_layers]

        self.vgg = tf.keras.Model(
            [vgg.input], [content_outputs, style_outputs]
        )
        self.vgg.trainable = False

    def call(self, inputs):
        preprocessed_input = vgg19.preprocess_input(inputs)
        content_outputs, style_outputs = self.vgg(preprocessed_input)
        return content_outputs, style_outputs
