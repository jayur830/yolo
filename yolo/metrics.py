import tensorflow as tf


class MeanAveragePrecision(tf.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, *args, **kwargs):
        pass

    def result(self):
        pass
