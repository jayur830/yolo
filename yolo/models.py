import tensorflow as tf

from yolo.losses import YOLOLoss


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
            loss=YOLOLoss(loss),
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            **kwargs)

    # def call(self, inputs, training=None, mask=None):
    #     if training is None or training == True:
    #         return inputs
    #     elif training == False:
    #         return tf.sigmoid(inputs)
    #     else:
    #         raise TypeError(f"The type of argument 'training' must be bool or None: {training}")
    #
    # def __call__(self, *args, **kwargs):
    #     return args
