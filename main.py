import os

from yolo.datasets import YOLODataset
from yolo.callbacks import YOLOLiveView
from model import model

if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    batch_size = 2
    epochs = 500

    dataset = YOLODataset(
        target_size=(128, 512),
        grid_size=(16, 64))
    # classes, x_train, y_train, x_test, y_test = dataset.flow_from_directory(directory="D:/Dataset/loon_rpn_split")
    classes, x_train, y_train, x_test, y_test = dataset.flow_from_directory(directory="D:/Dataset/image/loon_rpn_split")

    yolo = model(len(classes), learning_rate=1e-4)

    yolo.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=.2,
        callbacks=[
            YOLOLiveView(
                x=x_train,
                model=yolo,
                batch_size=batch_size,
                step_interval=40,
                target_size=dataset.target_size,
                grid_size=dataset.grid_size)
        ])
