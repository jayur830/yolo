import tensorflow as tf

from yolo import YOLO


def model(classes_len: int, learning_rate: float = 1e-2):
    # (128, 512, 3)
    input_layer = tf.keras.layers.Input(shape=(128, 512, 3))
    # (128, 512, 3) -> (64, 256, 8)
    x = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer="he_normal",
        use_bias=False)(input_layer)
    x = tf.keras.layers.BatchNormalization(momentum=.9)(x)
    x = tf.keras.layers.LeakyReLU(alpha=1e-2)(x)
    # (64, 256, 8) -> (32, 128, 16)
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer="he_normal",
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=.9)(x)
    x = tf.keras.layers.LeakyReLU(alpha=1e-2)(x)
    # (32, 128, 16) -> (16, 64, 32)
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer="he_normal",
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=.9)(x)
    x = tf.keras.layers.LeakyReLU(alpha=1e-2)(x)
    # (16, 64, 32) -> (8, 32, 64)
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer="he_normal",
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=.9)(x)
    x = tf.keras.layers.LeakyReLU(alpha=1e-2)(x)
    # (8, 32, 64) -> (4, 16, 128)
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer="he_normal",
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=.9)(x)
    x = tf.keras.layers.LeakyReLU(alpha=1e-2)(x)
    # (4, 16, 128) -> (4, 16, 64)
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=1,
        kernel_initializer="he_normal",
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=.9)(x)
    x = tf.keras.layers.LeakyReLU(alpha=1e-2)(x)
    # (4, 16, 64) -> (4, 16, 5 + classes_len)
    x = tf.keras.layers.Conv2D(
        filters=5 + classes_len,
        kernel_size=1,
        kernel_initializer="he_normal",
        use_bias=False)(x)
    x = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(x)

    yolo = YOLO(input_layer, x)
    yolo.compile(
        optimizer=tf.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=.9,
            nesterov=True))
    yolo.summary()

    return yolo