# Copyright (c) OpenMMLab. All rights reserved.
import base64
import mmcv
import os
import torch
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from ts.torch_handler.base_handler import BaseHandler

from mmdet.apis import inference_detector, init_detector, slided_inference_detector

MEAN_DOOR_SIZE = 40.82 # found out by averaging over all gt doors

def is_walking_direction_horizonal(staircase_patch):
    """
    Helper function to determine the walking direction, given a patch of a staircase and boxes
    from staircase detection. This is then used to determine the width of a staircase.
    """
    img = cv2.threshold(staircase_patch, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    diff_stairs_x = np.diff(img, axis=-1).astype(bool)
    diff_stairs_y = np.diff(img, axis=0).astype(bool)
    stairs_count_x = np.mean(np.sum(diff_stairs_x, axis=1))
    stairs_count_y = np.mean(np.sum(diff_stairs_y, axis=0))

    return stairs_count_y < stairs_count_x

def width_line_from_box(box, is_horizontal):
    """
    Helper function to create a line representing the width of a stair case, i.e. orthogonal to
    walking direction.
    """
    if is_horizontal:
        return [(box[0], (box[1] + box[3])/2), (box[2], (box[1] + box[3])/2)]
    else:
        return [((box[0] + box[2]) / 2, box[1]), ((box[0] + box[2]) / 2, box[3])]

class MMdetHandler(BaseHandler):
    threshold = 0.7
    SLIDED_INFERENCE_THRESHOLD = 40_000_000

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')

        self.model = init_detector(self.config_file, checkpoint, self.device)
        self.initialized = True

        self.image = None

    def preprocess(self, data):
        images = []

        for row in data:
            image = row.get('data') or row.get('body')
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = mmcv.imfrombytes(image)
            images.append(image)

        self.image = images[0]
        return images

    def inference(self, data, *args, **kwargs):

        if len(data) > 1:
            results = inference_detector(self.model, data)
        elif len(data) == 1:
            # handle large-scale drawings with slided inference.
            img_w = data[0].shape[0]
            img_h = data[0].shape[1]
            # large drawings
            if img_w * img_h > self.SLIDED_INFERENCE_THRESHOLD:
                results = slided_inference_detector(self.model,
                                                    data[0],
                                                    slide_size=(1792, 1792),
                                                    chip_size=(2048, 2048))
                # NOTE: HACK
                results = [results]
            # small drawings
            else:
                results = inference_detector(self.model, data)
        return results

    def postprocess(self, data):
        # Format output following the example ObjectDetectionHandler format
        output = []
        for image_index, image_result in enumerate(data):
            output.append([])
            if isinstance(image_result, tuple):
                bbox_result, segm_result = image_result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]  # ms rcnn
            else:
                bbox_result, segm_result = image_result, None

            for staircase in bbox_result[0]: # index 0 because only 1 class.
                stairs_patch = self.image[int(staircase[1]):int(staircase[3]),
                                          int(staircase[0]): int(staircase[2]),
                                          0]
                staircase_is_horizontal = detect_walking_direction(stairs_patch)

                bbox_coords = bbox[:-1].tolist()
                score = float(bbox[-1])
                width_line = width_line_from_box(bbox, staircase_is_horizontal)

                if score >= self.threshold:
                    output[image_index].append({
                        'bbox': bbox_coords,
                        'width_line': width_line,
                        'score': score
                    })

        return output
