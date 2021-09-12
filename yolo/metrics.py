import tensorflow as tf


class F1Score(tf.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, *args, **kwargs):
        pass

    def result(self):
        pass


class MeanAveragePrecision(tf.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, *args, **kwargs):
        pass

    def result(self):
        pass


class Recall(tf.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__recall = 0

    def update_state(self, *args, **kwargs):
        print(args)

    def result(self):
        return self.__recall


class Precision(tf.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, *args, **kwargs):
        pass

    def result(self):
        pass
