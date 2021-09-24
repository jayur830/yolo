import tensorflow as tf

from yolo.losses import YOLOLoss


class YOLO(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__input_layer, self.__output_layer = args[0], args[1]

    def compile(self,
                optimizer="rmsprop",
                loss="sse",
                iou_logits=False,
                lambda_coord=5.,
                lambda_noobj=.5,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                **kwargs):
        super(YOLO, self).compile(
            optimizer=optimizer,
            loss=YOLOLoss(loss, iou_logits, lambda_coord, lambda_noobj),
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            **kwargs)

    # def call(self, inputs, training=None, mask=None):
    #     output = super(YOLO, self).call(inputs, training, mask)
    #     return output if training else self.__convert(output)
    #
    # def __convert(self, inputs):
    #     return tf.sigmoid(inputs)
