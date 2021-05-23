from bs4 import BeautifulSoup
import numpy as np
from os import path

from .config import VOC_CLASSES


def parse_annotations(xml_path, keep_difficult=False, return_difficulty=False):
    with open(xml_path, 'r') as f:
        annotation = BeautifulSoup(f, 'xml')

    bboxes = []

    for obj in annotation.find_all('object'):
        difficulty = int(obj.difficult.text)

        if difficulty and not keep_difficult:
            continue

        label = obj.find('name').text
        xmin = float(obj.bndbox.xmin.text) - 1
        ymin = float(obj.bndbox.ymin.text) - 1
        xmax = float(obj.bndbox.xmax.text) - 1
        ymax = float(obj.bndbox.ymax.text) - 1
        box = [xmin, ymin, xmax, ymax, VOC_CLASSES.index(label)]

        if return_difficulty:
            box.append(difficulty)
        bboxes.append(box)

    return np.array(bboxes, np.int32)


def load_voc_dataset(dataroot='./data/VOCdevkit',
                     splits=[('VOC2007', 'trainval'), ('VOC2012', 'trainval')],
                     keep_difficult=False,
                     return_difficulty=False):
    img_paths, bboxes = [], []
    for year, split in splits:
        ids_file = path.join(
            dataroot, year, 'ImageSets', 'Main', split + '.txt')
        with open(ids_file, 'r') as f:
            ids = [line.strip() for line in f.readlines()]

        img_paths += [path.join(dataroot, year, 'JPEGImages', id+'.jpg')
                      for id in ids]
        bboxes += [parse_annotations(path.join(dataroot, year,
                                               'Annotations', id+'.xml'),
                                     keep_difficult=keep_difficult,
                                     return_difficulty=return_difficulty)
                   for id in ids]
    return np.array(img_paths), np.array(bboxes)
