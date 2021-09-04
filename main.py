import tensorflow as tf
import cv2

from data import YOLODataset
from model import model

batch_size = 2
epochs = 100

if __name__ == '__main__':
    dataset = YOLODataset()
    classes, x, y = dataset.flow_from_directory("D:/Dataset/loon_rpn_split")

    model = model(len(classes))

    def on_batch_end(batch, _logs):
        pass

    model.fit(
        x=x,
        y=y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=.2,
        callbacks=[tf.keras.callbacks.LambdaCallback(on_batch_end=on_batch_end)])
