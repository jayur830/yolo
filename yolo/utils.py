import tensorflow as tf
import numpy as np


def intersection_over_union(a: [int, int, int, int], b: [int, int, int, int]) -> float:
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])

    x1_max = max(a[0], b[0])
    x2_min = min(a[2], b[2])
    y1_max = max(a[1], b[1])
    y2_min = min(a[3], b[3])

    intersection = (x2_min - x1_max) * (y2_min - y1_max)

    try:
        return intersection / (a_area + b_area - intersection)
    except ZeroDivisionError:
        return 0.


def iou(a: [int, int, int, int], b: [int, int, int, int]) -> float:
    return intersection_over_union(a, b)


def tf_intersection_over_union(a, b):
    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])

    x1_max = tf.maximum(a[0], b[0])
    x2_min = tf.minimum(a[2], b[2])
    y1_max = tf.maximum(a[1], b[1])
    y2_min = tf.minimum(a[3], b[3])

    intersection = (x2_min - x1_max) * (y2_min - y1_max)

    try:
        return intersection / (a_area + b_area - intersection)
    except ZeroDivisionError:
        return tf.constant(0)


def tf_iou(a, b):
    return tf_intersection_over_union(a, b)


def non_max_suppression(bboxes: [[[int, int, int, int]]], iou_threshold: float = .45) -> [[int, int, int, int]]:
    final_boxes = []
    for bbox_class in bboxes:
        while len(bbox_class) > 0:
            compare_box = bbox_class[0]
            _final_boxes = [bbox_class[0]]
            bbox_class.remove(compare_box)
            for bbox in bbox_class:
                if iou(compare_box, bbox) >= iou_threshold:
                    _final_boxes.append(bbox)
                    bbox_class.remove(bbox)
            final_boxes += _final_boxes
    return final_boxes


def nms(bboxes: [[[int, int, int, int]]], iou_threshold: float = .45) -> [[int, int, int, int]]:
    return non_max_suppression(bboxes, iou_threshold)


# @tf.function
def confusion_matrix(y_true, y_pred, num_classes: int, conf_threshold: float = .5, iou_threshold: float = .45):
    y_true_shape = tf.shape(y_true)

    tp, fp, fn = \
        [0] * num_classes, \
        [0] * num_classes, \
        [0] * num_classes
    y_true_bboxes, y_pred_bboxes = \
        [[] for _ in range(num_classes)], \
        [[] for _ in range(num_classes)]

    @tf.function
    def while_loop_case_body_y_true(n, cy, cx):
        x = (y_true[n, cy, cx, 0] + tf.cast(cx, dtype=tf.float32)) / tf.cast(tf.shape(y_true)[2], dtype=tf.float32)
        y = (y_true[n, cy, cx, 1] + tf.cast(cy, dtype=tf.float32)) / tf.cast(tf.shape(y_true)[1], dtype=tf.float32)
        w = y_true[n, cy, cx, 2]
        h = y_true[n, cy, cx, 3]
        class_index = tf.argmax(y_true[n, cy, cx, 5:], axis=-1, output_type=tf.int32)

        for i in range(num_classes):
            tf.case([(
                tf.equal(tf.constant(i), class_index),
                lambda: y_true_bboxes[i].append([
                    x - w / 2,
                    y - h / 2,
                    x + w / 2,
                    y + h / 2
                ])
            )], exclusive=True)

    @tf.function
    def while_loop_case_body_y_pred(n, cy, cx):
        x = (y_pred[n, cy, cx, 0] + tf.cast(cx, dtype=tf.float32)) / tf.cast(tf.shape(y_pred)[2], dtype=tf.float32)
        y = (y_pred[n, cy, cx, 1] + tf.cast(cy, dtype=tf.float32)) / tf.cast(tf.shape(y_pred)[1], dtype=tf.float32)
        w = y_pred[n, cy, cx, 2]
        h = y_pred[n, cy, cx, 3]
        class_index = tf.argmax(y_pred[n, cy, cx, 5:], axis=-1, output_type=tf.int32)

        for i in range(num_classes):
            tf.case([(
                tf.equal(tf.constant(i), class_index),
                lambda: y_pred_bboxes[i].append([
                    x - w / 2,
                    y - h / 2,
                    x + w / 2,
                    y + h / 2
                ])
            )], exclusive=True)

    @tf.function
    def while_loop_body(n, cy, cx):
        tf.case([(
            tf.greater_equal(y_true[n, cy, cx, 4], conf_threshold),
            lambda: while_loop_case_body_y_true(n, cy, cx)
        )])
        tf.case([(
            tf.greater_equal(y_pred[n, cy, cx, 4], conf_threshold),
            lambda: while_loop_case_body_y_pred(n, cy, cx)
        )])
        return 0

    tf.while_loop(
        lambda n: tf.less(n, y_true_shape[0]),
        lambda n: tf.while_loop(
            lambda cy: tf.less(cy, y_true_shape[1]),
            lambda cy: tf.while_loop(
                lambda cx: tf.less(cx, y_true_shape[2]),
                lambda cx: while_loop_body(n, cy, cx),
                [0]),
            [0]),
        [0])

    # def argmax(arr):
    #     max_index, max = -1, -99999999999.9
    #     for i in range(len(arr)):
    #         if max < arr[i]:
    #             max = arr[i]
    #             max_index = i
    #     return max_index
    #
    # for n in range(len(y_true)):
    #     for cy in range(len(y_true[n])):
    #         for cx in range(len(y_true[n][cy])):
    #             if y_true[n][cy][cx][4] >= conf_threshold:
    #                 x, y, w, h = \
    #                     (y_true[n][cy][cx][0] + cx) / len(y_true[n][cy]), \
    #                     (y_true[n][cy][cx][1] + cy) / len(y_true[n]), \
    #                     y_true[n][cy][cx][2], \
    #                     y_true[n][cy][cx][3]
    #                 y_true_bboxes[argmax(y_true[n][cy][cx][5:])].append([
    #                     x - w / 2,
    #                     y - h / 2,
    #                     x + w / 2,
    #                     y + h / 2
    #                 ])
    #             if y_pred[n][cy][cx][4] >= conf_threshold:
    #                 x, y, w, h = \
    #                     (y_pred[n][cy][cx][0] + cx) / len(y_pred[n][cy]), \
    #                     (y_pred[n][cy][cx][1] + cy) / len(y_pred[n]), \
    #                     y_pred[n][cy][cx][2], \
    #                     y_pred[n][cy][cx][3]
    #                 y_pred_bboxes[argmax(y_pred[n][cy][cx][5:])].append([
    #                     x - w / 2,
    #                     y - h / 2,
    #                     x + w / 2,
    #                     y + h / 2
    #                 ])

    def add_tp(_exist, class_idx):
        _exist = True
        tp[class_idx] += 1
    def add_fp(class_idx):
        fp[class_idx] += 1

    for class_index in range(num_classes):
        num_class_objs = len(y_true_bboxes[class_index])
        for y_pred_bbox in y_pred_bboxes[class_index]:
            exist = False
            for y_true_bbox in y_true_bboxes[class_index]:
                iou_val = tf_iou(y_true_bbox[:4], y_pred_bbox[:4])
                iou_threshold_tensor = tf.constant(iou_threshold)
                print(f"iou_val: {iou_val}")
                print(f"iou_threshold: {iou_threshold}")
                constraint = tf.greater_equal(tf_iou(y_true_bbox[:4], y_pred_bbox[:4]), tf.constant(iou_threshold))
                print(f"constraint: {constraint}")
                tf.case([(
                    constraint,
                    lambda: add_tp(exist, class_index)
                )])
            tf.case([(
                not exist,
                lambda: add_fp(class_index)
            )])
        fn[class_index] = num_class_objs - tp[class_index]

    return tp, fp, fn


