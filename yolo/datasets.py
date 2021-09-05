import numpy as np
import cv2

from glob import glob
from concurrent.futures import ThreadPoolExecutor


class YOLODataset:
    def __init__(self, target_size: (int, int), grid_size: (int, int)):
        self.target_size, self.grid_size = target_size, grid_size

    def flow_from_directory(self, directory: str):
        directory = directory.replace("\\", "/")
        if directory[-1] != "/":
            directory += "/"

        # classes
        with open(directory + "classes.txt", "r") as reader:
            classes = [line.replace("\n", "") for line in reader.readlines()]

        # data
        thread_pool_executor = ThreadPoolExecutor(max_workers=16)
        x_data, y_data = [], []

        y_paths = glob(directory + "*.txt")

        futures = []
        for _path in y_paths:
            if "classes" in _path:
                continue
            futures.append(thread_pool_executor.submit(self.__load, _path, len(classes), x_data, y_data))
        for future in futures:
            future.result()

        indexes = np.arange(len(x_data))
        np.random.shuffle(indexes)
        x_data, y_data = np.asarray(x_data), np.asarray(y_data)
        x_data, y_data = x_data[indexes], y_data[indexes]

        return classes, x_data, y_data

    def __load(self, path: str, classes_len: int, x_data: [], y_data: []):
        path = path.replace("\\", "/")

        # X
        img = cv2.resize(
            src=cv2.imread(path.replace(".txt", "") + ".jpg"),
            dsize=(self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_AREA)
        x_data.append(img)

        # Y
        label_tensor = np.zeros(shape=self.grid_size + (5 + classes_len,))
        with open(path, "r") as reader:
            label = [line.replace("\n", "").split(" ") for line in reader.readlines()]

        for l in label:
            class_index, x, y, w, h = (int(l[0]), float(l[1]), float(l[2]), float(l[3]), float(l[4]))
            grid_x, grid_y, x, y, w, h = self.__to_yolo_format(self.grid_size[1], self.grid_size[0], x, y, w, h)
            label_tensor[grid_y, grid_x, 0] = x
            label_tensor[grid_y, grid_x, 1] = y
            label_tensor[grid_y, grid_x, 2] = w
            label_tensor[grid_y, grid_x, 3] = h
            label_tensor[grid_y, grid_x, 4] = 1.
            label_tensor[grid_y, grid_x, 5 + class_index] = 1.
        y_data.append(label_tensor)

    def __to_yolo_format(self, grid_width: int, grid_height: int, x: float, y: float, w: float, h: float):
        grid_x, grid_y = int(x * grid_width), int(y * grid_height)
        x, y = x * grid_width - grid_x, y * grid_height - grid_y
        return grid_x, grid_y, x, y, w, h

    @staticmethod
    def convert(tensor, target_size: (int, int), grid_size: (int, int), conf_threshold: float = .5) -> [[int, int, int, int]]:
        bboxes = []
        for batch in range(tensor.shape[0]):
            for height in range(tensor.shape[1]):
                for width in range(tensor.shape[2]):
                    if tensor[batch, height, width, 4] >= conf_threshold:
                        grid_x, grid_y, x, y, w, h = width, height, tensor[batch, height, width, 0], tensor[batch, height, width, 1], tensor[batch, height, width, 2], tensor[batch, height, width, 3]
                        x = target_size[1] * (grid_x + x) / grid_size[1]
                        y = target_size[0] * (grid_y + y) / grid_size[0]
                        w *= target_size[1]
                        h *= target_size[0]
                        bboxes.append([int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)])
        return bboxes
