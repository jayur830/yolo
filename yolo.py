import tensorflow as tf


class YOLO(tf.keras.models.Model):
    def compile(self,
              optimizer='rmsprop',
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              **kwargs):
        super(YOLO, self).compile(
            optimizer=optimizer,
            loss=self.__yolo_loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            **kwargs)

    def __yolo_loss(self, y_true, y_pred):
        pass
