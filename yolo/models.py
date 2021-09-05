import tensorflow as tf


class YOLO(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compile(self,
                optimizer="rmsprop",
                loss="sse",
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                **kwargs):
        super(YOLO, self).compile(
            optimizer=optimizer,
            loss=self.__loss(loss),
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            **kwargs)

    def __loss(self, loss_function):
        if type(loss_function) == str:
            if loss_function == "sse" or loss_function == "sum_squared" or loss_function == "sum_squared_error":
                loss_function = lambda a, b: tf.reduce_sum(tf.square(a - b))
            elif loss_function == "rsse" or loss_function == "root_sum_squared" or loss_function == "root_sum_squared_error":
                loss_function = lambda a, b: tf.sqrt(tf.reduce_sum(tf.square(a - b)))
            elif loss_function == "sae" or loss_function == "sum_absolute" or loss_function == "sum_absolute_error":
                loss_function = lambda a, b: tf.reduce_sum(tf.abs(a - b))
            elif loss_function == "rsse" or loss_function == "root_sum_absolute" or loss_function == "root_sum_absolute_error":
                loss_function = lambda a, b: tf.sqrt(tf.reduce_sum(tf.abs(a - b)))
            elif loss_function == "mse" or loss_function == "mean_squared" or loss_function == "mean_squared_error":
                loss_function = tf.losses.mean_squared_error
            elif loss_function == "rmse" or loss_function == "root_mean_squared" or loss_function == "root_mean_squared_error":
                loss_function = lambda a, b: tf.sqrt(tf.losses.mean_squared_error(a, b))
            elif loss_function == "mae" or loss_function == "mean_absolute" or loss_function == "mean_absolute_error":
                loss_function = tf.losses.mean_absolute_error
            elif loss_function == "rmae" or loss_function == "root_mean_absolute" or loss_function == "root_mean_absolute_error":
                loss_function = lambda a, b: tf.sqrt(tf.losses.mean_absolute_error(a, b))
            elif loss_function == "bce" or loss_function == "binary_crossentropy":
                loss_function = tf.losses.binary_crossentropy
            elif loss_function == "hinge":
                loss_function = tf.losses.hinge
            elif loss_function == "kld" or loss_function == "kl_divergence":
                loss_function = tf.losses.kl_divergence
            else:
                raise ValueError(f"Could not interpret loss function identifier: {loss_function}")

        def _loss(y_true, y_pred):
            y_pred = tf.sigmoid(y_pred)
            lambda_coord, lambda_noobj, eps = 5., .5, 1e-16

            xy_true, xy_pred = y_true[:, :, :, 0:2], y_pred[:, :, :, 0:2]
            wh_true, wh_pred = y_true[:, :, :, 2:4], y_pred[:, :, :, 2:4]
            conf_true, conf_pred = tf.expand_dims(y_true[:, :, :, 4], axis=-1), tf.expand_dims(y_pred[:, :, :, 4], axis=-1)
            class_true, class_pred = y_true[:, :, :, 5:], y_pred[:, :, :, 5:]

            xy_loss = lambda_coord * loss_function(xy_true, xy_pred * conf_true)
            wh_loss = lambda_coord * loss_function((wh_true + eps) ** .5, (wh_pred * conf_true + eps) ** .5)
            conf_loss = loss_function(conf_true, conf_pred * conf_true) + lambda_noobj * loss_function(0., conf_pred * tf.cast(tf.logical_not(tf.cast(conf_true, dtype=tf.bool)), dtype=tf.float32))
            class_loss = loss_function(class_true, class_pred * conf_true)

            return xy_loss + wh_loss + conf_loss + class_loss
        return _loss
