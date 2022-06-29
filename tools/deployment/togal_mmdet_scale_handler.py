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

def cluster_dimensions(result_bboxes):
    ss = StandardScaler()
    boxes = np.array([box['bbox'] for box in result_bboxes])
    bbox_dims_x = boxes[:, 2] - boxes[:, 0]
    bbox_dims_y = boxes[:, 3] - boxes[:, 1]
    X = np.vstack([bbox_dims_x, bbox_dims_y]).reshape((len(bbox_dims_x), 2))
    X_scaled = ss.fit_transform(X)
    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=0.9, min_samples=2).fit(X_scaled)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    cluster_labels = np.delete(labels, np.where(labels == -1))
    cluster_means_scaled = []

    for cluster_idx in np.unique(cluster_labels):
        cluster_means_scaled.append(np.mean(X_scaled[np.where(labels == cluster_idx)[0]], axis=0))

    cluster_means = ss.inverse_transform(cluster_means_scaled)
    cluster_means = np.array(cluster_means)

    print("Cluster means: ", cluster_means)
    # aggregate like a madman
    return cluster_means


class MMdetHandler(BaseHandler):
    threshold = 0.3
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

    def preprocess(self, data):
        images = []

        for row in data:
            image = row.get('data') or row.get('body')
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = mmcv.imfrombytes(image)
            images.append(image)

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
                                                    data,
                                                    slide_size=(1792, 1792),
                                                    chip_size=(2048, 2048))
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

            for class_index, class_result in enumerate(bbox_result):
                class_name = self.model.CLASSES[class_index]
                for bbox in class_result:
                    bbox_coords = bbox[:-1].tolist()
                    score = float(bbox[-1])
                    if score >= self.threshold:
                        output[image_index].append({
                            'class_name': class_name,
                            'bbox': bbox_coords,
                            'score': score
                        })

        dimension_clusters = cluster_dimensions(output[0])

        return {'cluster_means': dimension_clusters, 'scale_factor': np.mean(dimension_clusters) / MEAN_DOOR_SIZE}
