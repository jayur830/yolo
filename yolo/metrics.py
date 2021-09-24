import tensorflow as tf
import numpy as np

from yolo.utils import confusion_matrix


class F1Score(tf.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(name="f1", **kwargs)

    def update_state(self, *args, **kwargs):
        pass

    def result(self):
        pass


class MeanAveragePrecision(tf.metrics.Metric):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(name="mAP", **kwargs)
        self._num_classes = num_classes
        self._recall_val, self._precision_val, self.__mAP = 0, 0, 0

    def update_state(self, *args, **kwargs):
        y_true, y_pred = args[0], args[1]

    def result(self):
        return self.__mAP

    def _recall(self, y_true, y_pred):
        tp, _, fn = confusion_matrix(y_true, y_pred, self._num_classes)
        for i in range(self._num_classes):
            self._recall_val += tp / (tp + fn)
        self._recall_val /= self._num_classes

    def _precision(self, y_true, y_pred):
        tp, fp, _ = confusion_matrix(y_true, y_pred, self._num_classes)
        for i in range(self._num_classes):
            self._precision_val += tp / (tp + fp)
        self._precision_val /= self._num_classes


class Recall(MeanAveragePrecision):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes, **kwargs)

    def update_state(self, *args, **kwargs):
        y_true, y_pred = args[0], args[1]
        self._recall(y_true, y_pred)

    def result(self):
        return self._recall_val


class Precision(MeanAveragePrecision):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes, **kwargs)

    def update_state(self, *args, **kwargs):
        y_true, y_pred = args[0], args[1]
        self._precision(y_true, y_pred)

    def result(self):
        return self._precision_val
