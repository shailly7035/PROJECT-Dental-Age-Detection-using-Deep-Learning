# import tensorflow as tf
# import numpy as np
# import cv2
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pandas as pd
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def object_detection(filepath):
    sess =tf.compat.v1.InteractiveSession()
    MODEL_NAME = 'inference_graph'
    CWD_PATH = os.path.join(os.getcwd(),'object_detection')
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
    NUM_CLASSES = 8
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)


    path = filepath
    for filename in (os.listdir(path)):
        if '.jpg' in filename:
            detected_boxes = []
            IMAGE_NAME = filename
            PATH_TO_IMAGE = os.path.join(path,IMAGE_NAME)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image = cv2.imread(PATH_TO_IMAGE)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_expanded = np.expand_dims(image, axis=0)
            h = image.shape[0]
            w = image.shape[1]
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})
            for i, box in enumerate(np.squeeze(boxes)):
                if (np.squeeze(scores)[i] > 0.60):
                    box[0] = int(box[0] * h)
                    box[1] = int(box[1] * w)
                    box[2] = int(box[2] * h)
                    box[3] = int(box[3] * w)
                    box = np.append(box, np.squeeze(classes)[i])
                    box = np.append(box, int(np.squeeze(scores)[i]*100))
                    detected_boxes.append(box)
                    
            return (detected_boxes)
        
