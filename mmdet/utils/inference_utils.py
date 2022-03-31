def delete_border_detections(chip_detections, chip_w, border_delete_amount):
    """Deletes detections near borders. This is to make merging of several slided inference patches easier.
    The function is implemented vectorized and is therefore lightning fast.

    Args:
        chip_detections (list): List of np.arrays of detections of a single patch.
        chip_w (int): Width of a patch.
        border_delete_amount (int): How much of the border should be deleted.

    Returns:
        chip_detections (list): Updated list of (smaller or equal-sized) np.arrays of detections of a single patch.
    """
    new_chip_detections = []
    x_min_allowed = border_delete_amount
    y_min_allowed = border_delete_amount
    x_max_allowed = chip_w - border_delete_amount
    y_max_allowed = chip_w - border_delete_amount
    for class_detections in chip_detections:
        class_detections = class_detections[class_detections[:, 0] > x_min_allowed]
        class_detections = class_detections[class_detections[:, 1] > y_min_allowed]
        class_detections = class_detections[class_detections[:, 2] < x_max_allowed]
        class_detections = class_detections[class_detections[:, 3] < y_max_allowed]
        new_chip_detections.append(class_detections)
    return new_chip_detections

def nms(bounding_boxes, confidence_score, threshold):
    """Vectorized implementation of non-maximum-suppression.

    Args:
        bounding_boxes (np.array,4): np.arrays of detections of a single class.
        confidence_score (np.array,1): Width of a patch.
        threshold (int): How much of the border should be deleted.

    Returns:
        chip_detections (list): Updated list of (smaller or equal-sized) np.arrays of detections of a single patch.
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        iou = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(iou < threshold)
        order = order[left]

    return picked_boxes, picked_score

