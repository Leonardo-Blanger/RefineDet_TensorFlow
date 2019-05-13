from bs4 import BeautifulSoup
import numpy as np
import os

voc_classes = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

def parse(xml_path, classes = voc_classes, keep_difficult = False, return_difficulty = False):
    if not os.path.exists(xml_path):
        raise Exception('Annotation file %s not found' % xml_path)

    with open(xml_path, 'r') as f:
        annotation = BeautifulSoup(f, 'xml')

    img_height = int(annotation.size.height.text)
    img_width = int(annotation.size.width.text)
    img_depth = int(annotation.size.depth.text)

    bnd_boxes = []

    for obj in annotation.find_all('object'):
        d = int(obj.difficult.text)

        if d == 0 or keep_difficult:
            label = obj.find('name').text
            xmin = float(obj.bndbox.xmin.text) - 1
            ymin = float(obj.bndbox.ymin.text) - 1
            xmax = float(obj.bndbox.xmax.text) - 1
            ymax = float(obj.bndbox.ymax.text) - 1

            if return_difficulty:
                bnd_boxes.append([xmin, ymin, xmax, ymax, classes.index(label), d])
            else:
                bnd_boxes.append([xmin, ymin, xmax, ymax, classes.index(label)])

    return np.array(bnd_boxes, np.float32)
