import tensorflow as tf

from tensorflow.python.framework.ops import convert_to_tensor_v2


class YOLOLoss(tf.keras.losses.Loss):
    def __init__(self,
                 loss,
                 lambda_coord: float = 5.,
                 lambda_noobj: float = .5):
        super(YOLOLoss, self).__init__()
        if type(loss) == str:
            if loss == "sse" or loss == "sum_squared" or loss == "sum_squared_error":
                loss = lambda a, b, c: tf.reduce_sum(tf.reduce_sum(tf.square(a - b), axis=-1) * c)
            elif loss == "rsse" or loss == "root_sum_squared" or loss == "root_sum_squared_error":
                loss = lambda a, b, c: tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=-1) * c))
            elif loss == "sae" or loss == "sum_absolute" or loss == "sum_absolute_error":
                loss = lambda a, b, c: tf.reduce_sum(tf.reduce_sum(tf.abs(a - b), axis=-1) * c)
            elif loss == "rsae" or loss == "root_sum_absolute" or loss == "root_sum_absolute_error":
                loss = lambda a, b, c: tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.abs(a - b), axis=-1) * c))
            elif loss == "mse" or loss == "mean_squared" or loss == "mean_squared_error":
                loss = lambda a, b, c: tf.reduce_mean(tf.reduce_mean(tf.square(a - b), axis=-1) * c)
            elif loss == "rmse" or loss == "root_mean_squared" or loss == "root_mean_squared_error":
                loss = lambda a, b, c: tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(a - b), axis=-1) * c))
            elif loss == "mae" or loss == "mean_absolute" or loss == "mean_absolute_error":
                loss = lambda a, b, c: tf.reduce_mean(tf.reduce_mean(tf.abs(a - b), axis=-1) * c)
            elif loss == "rmae" or loss == "root_mean_absolute" or loss == "root_mean_absolute_error":
                loss = lambda a, b, c: tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.abs(a - b), axis=-1) * c))
            elif loss == "bce" or loss == "binary_crossentropy":
                loss = lambda a, b, c: tf.reduce_mean(-tf.reduce_mean(a * tf.math.log(b) + (1 - a) * tf.math.log(1 - b), axis=-1) * c)
            elif loss == "hinge":
                loss = lambda a, b, c: tf.losses.hinge(a, b) * c
            elif loss == "kld" or loss == "kl_divergence":
                loss = tf.losses.kl_divergence
            else:
                raise ValueError(f"Could not interpret loss function identifier: {loss}")

        self.__loss = loss
        self.__lambda_coord = lambda_coord
        self.__lambda_noobj = lambda_noobj
        self.__conf_true = None

    def call(self, y_true, y_pred):
        y_pred = tf.sigmoid(convert_to_tensor_v2(y_pred))
        y_true = tf.cast(y_true, y_pred.dtype)
        self.__conf_true = y_true[:, :, :, 4]

        return self.__localization_loss(y_true, y_pred) + \
            self.__confidence_loss(y_true, y_pred) + \
            self.__classification_loss(y_true, y_pred)

    def __localization_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        xy_loss = self.__lambda_coord * self.__loss(
            y_true[:, :, :, 0:2],
            y_pred[:, :, :, 0:2],
            self.__conf_true)
        wh_loss = self.__lambda_coord * self.__loss(
            tf.sqrt(y_true[:, :, :, 2:4]),
            tf.sqrt(y_pred[:, :, :, 2:4]),
            self.__conf_true)
        return xy_loss + wh_loss


    def __confidence_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        return self.__loss(
            tf.expand_dims(y_true[:, :, :, 4], axis=-1),
            tf.expand_dims(y_pred[:, :, :, 4], axis=-1),
            tf.where(
                tf.cast(self.__conf_true, dtype=tf.bool),
                tf.ones(shape=tf.shape(input=self.__conf_true)),
                tf.ones(shape=tf.shape(input=self.__conf_true)) * self.__lambda_noobj))

    def __classification_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        return self.__loss(
            y_true[:, :, :, 5:],
            y_pred[:, :, :, 5:],
            self.__conf_true)
