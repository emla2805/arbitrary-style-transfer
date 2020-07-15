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


class TransferNet(tf.keras.Model):
    def __init__(self, encoder, decoder, content_weight, style_weight):
        super(TransferNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.content_weight = content_weight
        self.style_weight = style_weight

    def encode(self, content_img, style_img, alpha):
        content_feat_content, _ = self.encoder(content_img)
        content_feat_style, _ = self.encoder(style_img)

        t = adaptive_instance_normalization(content_feat_content, content_feat_style)
        t = alpha * t + (1 - alpha) * content_feat_content
        return t

    def call(self, data, alpha=1.0):
        content_img, style_img = data
        t = self.encode(content_img, style_img, alpha=alpha)
        g_t = self.decoder(t)
        return g_t

    def compile(self, optimizer, content_loss_fn, style_loss_fn):
        super(TransferNet, self).compile()
        self.optimizer = optimizer
        self.content_loss_fn = content_loss_fn
        self.style_loss_fn = style_loss_fn

    def train_step(self, data):
        content_img, style_img = data
        alpha = 1.0

        content_feat_content, style_feat_content = self.encoder(content_img)
        content_feat_style, style_feat_style = self.encoder(style_img)

        t = adaptive_instance_normalization(content_feat_content, content_feat_style)
        t = alpha * t + (1 - alpha) * content_feat_content

        with tf.GradientTape() as tape:
            stylized_img = self.decoder(t)
            content_feat_stylized, style_feat_stylized = self.encoder(stylized_img)

            style_loss = self.style_weight * self.style_loss_fn(
                style_feat_style, style_feat_stylized
            )
            content_loss = self.content_weight * self.content_loss_fn(
                t, content_feat_stylized
            )
            loss = style_loss + content_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )
        return {
            "loss/total": loss,
            "loss/style": style_loss,
            "loss/content": content_loss,
        }


def adaptive_instance_normalization(content_feat, style_feat, epsilon=1e-5):
    content_mean, content_variance = tf.nn.moments(
        content_feat, axes=[1, 2], keepdims=True
    )
    style_mean, style_variance = tf.nn.moments(
        style_feat, axes=[1, 2], keepdims=True
    )
    style_std = tf.math.sqrt(style_variance + epsilon)

    norm_content_feat = tf.nn.batch_normalization(
        content_feat,
        mean=content_mean,
        variance=content_variance,
        offset=style_mean,
        scale=style_std,
        variance_epsilon=epsilon,
    )
    return norm_content_feat


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

        content_output = vgg.get_layer(content_layer).output
        style_outputs = [vgg.get_layer(name).output for name in style_layers]

        self.vgg = tf.keras.Model([vgg.input], [content_output, style_outputs])
        self.vgg.trainable = False

    def call(self, inputs):
        preprocessed_input = vgg19.preprocess_input(inputs)
        content_outputs, style_outputs = self.vgg(preprocessed_input)
        return content_outputs, style_outputs
