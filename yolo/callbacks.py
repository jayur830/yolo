import tensorflow as tf
import numpy as np
import cv2

from yolo.datasets import YOLODataset


class YOLOLiveView(tf.keras.callbacks.Callback):
    def __init__(self,
                 x,
                 model,
                 batch_size: int,
                 step_interval: int,
                 target_size: (int, int),
                 grid_size: (int, int)):
        super().__init__()
        self.__x = x
        self.__model = model
        self.__batch_size = batch_size
        self.__step_interval = step_interval
        self.__target_size = target_size
        self.__grid_size = grid_size

    def on_batch_end(self, batch, logs=None):
        if batch % self.__step_interval == 0:
            img = self.__x[batch * self.__batch_size].copy()
            output = np.asarray(self.__model(img.reshape((1,) + img.shape), training=False))
            for x1, y1, x2, y2 in YOLODataset.convert(output, self.__target_size, self.__grid_size):
                img = cv2.rectangle(
                    img=img,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=(0, 0, 255),
                    thickness=2)
            cv2.imshow("View", img)
            cv2.waitKey(1)
