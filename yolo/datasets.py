import numpy as np
import cv2
import os

from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from yolo.utils import nms

class YOLODataset:
    def __init__(self, target_size: (int, int), grid_size: (int, int)):
        self.target_size, self.grid_size = target_size, grid_size

    def flow_from_directory(self, directory: str, test_split: float = .2):
        directory = directory.replace("\\", "/")
        if directory[-1] != "/":
            directory += "/"

        # classes
        with open(directory + "classes.txt", "r") as reader:
            classes = [line.replace("\n", "") for line in reader.readlines()]

        # data
        data = []
        thread_pool_executor = ThreadPoolExecutor(max_workers=16)

        y_paths = glob(directory + "*.jpg")

        def load(path: str):
            path = path.replace("\\", "/")
            label_path = f"{path[:-4]}.txt"

            if os.path.exists(path) and os.path.exists(label_path):
                # X
                img = cv2.resize(
                    src=cv2.imread(path),
                    dsize=(self.target_size[1], self.target_size[0]),
                    interpolation=cv2.INTER_AREA)

                # Y
                label_tensor = np.zeros(shape=self.grid_size + (5 + len(classes),))
                with open(label_path, "rt") as reader:
                    label = [line.replace("\n", "").split(" ") for line in reader.readlines()]

                for l in label:
                    class_index, x, y, w, h = list(map(float, l))
                    grid_x, grid_y, x, y, w, h = self.__to_yolo_format(self.grid_size[1], self.grid_size[0], x, y, w, h)
                    label_tensor[grid_y, grid_x, 0] = x
                    label_tensor[grid_y, grid_x, 1] = y
                    label_tensor[grid_y, grid_x, 2] = w
                    label_tensor[grid_y, grid_x, 3] = h
                    label_tensor[grid_y, grid_x, 4] = 1.
                    label_tensor[grid_y, grid_x, 5 + int(class_index)] = 1.
                data.append([img, label_tensor])

        futures = []
        for _path in y_paths:
            futures.append(thread_pool_executor.submit(load, _path))
        for future in tqdm(futures):
            future.result()

        data = np.asarray(data, dtype=np.object).transpose()
        x_data, y_data = data[0], data[1]
        indexes = np.arange(len(x_data))
        np.random.shuffle(indexes)
        x_data, y_data = x_data[indexes], y_data[indexes]

        return classes, x_data[int(x_data.shape[0] * test_split):], y_data[int(y_data.shape[0] * test_split):], x_data[:int(x_data.shape[0] * test_split)], y_data[:int(y_data.shape[0] * test_split)]

    def __to_yolo_format(self, grid_width: int, grid_height: int, x: float, y: float, w: float, h: float):
        grid_x, grid_y = int(x * grid_width), int(y * grid_height)
        x, y = x * grid_width - grid_x, y * grid_height - grid_y
        return grid_x, grid_y, x, y, w, h

    @staticmethod
    def convert(tensor, target_size: (int, int), grid_size: (int, int), conf_threshold: float = .5, iou_threshold: float = .45) -> [[int, int, int, int]]:
        tensor = 1 / (1 + np.exp(-tensor))
        bboxes = [[] for _ in range(tensor.shape[-1] % 5)]
        for batch in range(tensor.shape[0]):
            for height in range(tensor.shape[1]):
                for width in range(tensor.shape[2]):
                    if tensor[batch, height, width, 4] >= conf_threshold:
                        grid_x, grid_y, x, y, w, h, conf, class_index = \
                            width, \
                            height, \
                            tensor[batch, height, width, 0], \
                            tensor[batch, height, width, 1], \
                            tensor[batch, height, width, 2], \
                            tensor[batch, height, width, 3], \
                            tensor[batch, height, width, 4], \
                            int(np.argmax(tensor[batch, height, width, 5:]))
                        x = target_size[1] * (grid_x + x) / grid_size[1]
                        y = target_size[0] * (grid_y + y) / grid_size[0]
                        w *= target_size[1]
                        h *= target_size[0]
                        bboxes[class_index].append([int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2), conf])
        for i, bbox_class in enumerate(bboxes):
            bboxes[i] = [_bbox_class[:-1] for _bbox_class in sorted(bbox_class, key=lambda bbox: bbox[4], reverse=True)]
        return nms(bboxes, iou_threshold)
