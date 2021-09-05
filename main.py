import os

from yolo.datasets import YOLODataset
from yolo.callbacks import YOLOLiveView
from model import model

if __name__ == '__main__':
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    batch_size = 2
    epochs = 500

    dataset = YOLODataset(
        target_size=(128, 512),
        grid_size=(16, 64))
    # classes, x, y = dataset.flow_from_directory(directory="D:/Dataset/loon_rpn_split")
    classes, x, y = dataset.flow_from_directory(directory="D:/Dataset/image/loon_rpn_split")

    model = model(len(classes), learning_rate=1e-4)

    model.fit(
        x=x,
        y=y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=.2,
        callbacks=[
            YOLOLiveView(
                x=x,
                model=model,
                batch_size=batch_size,
                step_interval=20,
                target_size=dataset.target_size,
                grid_size=dataset.grid_size)
        ])
