

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


def non_max_suppression(bboxes: [[[int, int, int, int]]], iou_threshold: float = .5) -> [[int, int, int, int]]:
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


def nms(bboxes: [[[int, int, int, int]]], iou_threshold: float = .5) -> [[int, int, int, int]]:
    return non_max_suppression(bboxes, iou_threshold)