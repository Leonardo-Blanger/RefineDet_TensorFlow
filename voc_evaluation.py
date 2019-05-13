import tensorflow as tf
tf.enable_eager_execution()

import argparse
import importlib
import os
import cv2
import numpy as np
from tqdm import tqdm

from parsers import voc_parser
from models import RefineDetVGG16
import metrics

def evaluate(load_weights, dataset_path, config = 'config/voc320.py', batch_size = None,
                num_parallel_calls = 4, prefetch_buffer = 100):

    if isinstance(config, str):
        # Get the configuration module
        config_module = '.'.join(config.split('/'))[:-3]
        config = importlib.import_module(config_module)

    model = RefineDetVGG16()
    model.build(input_shape = (None, config.image_size, config.image_size, 3))
    model.load_weights(load_weights)
    model.summary()

    ids_file = os.path.join(dataset_path, 'VOC2007/ImageSets/Main/test.txt')

    with open(ids_file, 'r') as f:
        ids = [line.strip() for line in f]

    image_paths = [
        os.path.join(dataset_path, 'VOC2007', 'JPEGImages', id+'.jpg')
        for id in ids
    ]
    annotation_paths = [
        os.path.join(dataset_path, 'VOC2007', 'Annotations', id+'.xml')
        for id in ids
    ]

    num_samples = len(image_paths)
    test_dataset = tf.data.Dataset.from_tensor_slices((image_paths, annotation_paths))

    image_size = config.image_size
    channel_means = config.channel_means
    if batch_size is None: batch_size = config.batch_size

    # Method for loading and preprocessing a single data sample
    def load(img_path, ann_path):
        img_path = img_path.numpy().decode("utf-8")
        ann_path = ann_path.numpy().decode("utf-8")

        image = cv2.imread(img_path).astype('float32')
        boxes = voc_parser.parse(ann_path, classes = config.voc_classes, keep_difficult = False)

        # Resize the image and boxes
        curr_height, curr_width = image.shape[:2]
        image = cv2.resize(image, (image_size, image_size))
        boxes[:, (0, 2)] *= image_size / curr_width
        boxes[:, (1, 3)] *= image_size / curr_height

        # Subtract the channel means
        image -= channel_means

        # From BGR to RGB
        image = image[..., (2,1,0)]

        return image, boxes

    test_dataset = test_dataset.map(
        lambda image, boxes: tf.py_function(
            load, [image, boxes], [tf.float32, tf.float32]
        ),
        num_parallel_calls = num_parallel_calls
    )

    test_dataset = test_dataset.padded_batch(batch_size,
        padded_shapes = ([image_size, image_size, 3], [None, 5]),
        drop_remainder = False
    )

    test_dataset = test_dataset.prefetch(prefetch_buffer)
    batch_iterator = iter(test_dataset)

    y_true = []
    y_pred = []

    for _ in tqdm(range(int(np.ceil(num_samples / batch_size))), 'Running evaluation on test set'):
        images, boxes_true = next(batch_iterator)

        boxes_true = [
            tf.convert_to_tensor([box for box in boxes if box[4].numpy() > 0])
            for boxes in boxes_true
        ]

        boxes_pred = model(images, decode = True)

        y_true += boxes_true
        y_pred += boxes_pred

    return y_true, y_pred



def build_dataset(dataset_path = '../VOCdevkit',
                 config = 'config/voc320.py',
                 num_parallel_calls = 4):

    if isinstance(config, str):
        # Get the configuration module
        config_module = '.'.join(config.split('/'))[:-3]
        config = importlib.import_module(config_module)

    ids_file = os.path.join(dataset_path, 'VOC2007/ImageSets/Main/test.txt')

    with open(ids_file, 'r') as f:
        ids = [line.strip() for line in f]

    image_paths = [
        os.path.join(dataset_path, 'VOC2007', 'JPEGImages', id+'.jpg')
        for id in ids
    ]
    annotation_paths = [
        os.path.join(dataset_path, 'VOC2007', 'Annotations', id+'.xml')
        for id in ids
    ]

    num_samples = len(image_paths)
    test_dataset = tf.data.Dataset.from_tensor_slices((image_paths, annotation_paths))

    image_size = config.image_size
    channel_means = config.channel_means

    # Method for loading and preprocessing a single data sample
    def load(img_path, ann_path):
        img_path = img_path.numpy().decode("utf-8")
        ann_path = ann_path.numpy().decode("utf-8")

        image = cv2.imread(img_path).astype('float32')
        boxes = voc_parser.parse(ann_path, classes = config.voc_classes,
                                 keep_difficult = True,
                                 return_difficulty = True)

        # Resize the image and boxes
        curr_height, curr_width = image.shape[:2]
        image = cv2.resize(image, (image_size, image_size))
        boxes[:, (0, 2)] *= image_size / curr_width
        boxes[:, (1, 3)] *= image_size / curr_height

        # Subtract the channel means
        image -= channel_means

        # From BGR to RGB
        image = image[..., (2,1,0)]

        return image, boxes

    test_dataset = test_dataset.map(
        lambda image, boxes: tf.py_function(
            load, [image, boxes], [tf.float32, tf.float32]
        ),
        num_parallel_calls = num_parallel_calls
    )

    return test_dataset, num_samples


def run_on_data(model, dataset, num_samples,
                config = 'config/voc320.py',
                batch_size = None,
                prefetch_buffer = 100):

    if isinstance(config, str):
        # Get the configuration module
        config_module = '.'.join(config.split('/'))[:-3]
        config = importlib.import_module(config_module)

    if batch_size is None:
        batch_size = config.batch_size
    image_size = config.image_size

    dataset = dataset.padded_batch(batch_size,
        padded_shapes = ([image_size, image_size, 3], [None, None]),
        drop_remainder = False
    )

    dataset = dataset.prefetch(prefetch_buffer)
    batch_iterator = iter(dataset)

    y_true = []
    y_pred = []

    for i in tqdm(range(int(np.ceil(num_samples / batch_size))), 'Running evaluation on test set'):
        images, boxes_true = next(batch_iterator)

        boxes_true = [
            tf.convert_to_tensor([box for box in boxes if box[4].numpy() > 0])
            for boxes in boxes_true
        ]

        boxes_pred = model(images, decode = True)

        y_true += boxes_true
        y_pred += boxes_pred

    return y_true, y_pred


def voc_evaluation(model_weights = 'RefineDetVOC_320_120000.h5',
                   config = 'config/voc320.py',
                   dataset_path = '../VOCdevkit',
                   batch_size = None,
                   num_parallel_calls = 4,
                   prefetch_buffer = 100):

    if isinstance(config, str):
        # Get the configuration module
        config_module = '.'.join(config.split('/'))[:-3]
        config = importlib.import_module(config_module)

    print('Building the model...')
    model = RefineDetVGG16(num_classes = len(config.voc_classes),
                           background_id = config.voc_classes.index('background'))
    model.build(input_shape = (None, config.image_size, config.image_size, 3))

    print('Loading weights...')
    model.load_weights(model_weights)
    model.summary()

    print('Initializing data loaders...')
    # Returns a tf.data.Dataset object
    dataset, num_samples = build_dataset(dataset_path, config, num_parallel_calls)

    print('Running the model on the test images...')
    y_true, y_pred = run_on_data(model, dataset, num_samples, config, batch_size, prefetch_buffer)

    print('Running evaluation...')
    APs = metrics.per_class_AP(y_true, y_pred, iou_threshold = 0.5, version = '07')

    for cls, AP in zip(config.voc_classes[1:], APs):
        print('%s AP: %.2f%c' % (cls, 100*AP, '%'))

    meanAP = np.mean(APs)
    print('meanAP: %.2f%c' % (100*meanAP, '%'))

    return meanAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'RefineDet detector evaluation on PASCAL VOC 07 test'
    )
    parser.add_argument('--model_weights', default = 'weights/RefineDetVOC_320_120000.h5', type = str,
                        help = 'Path to a saved weights file.')
    parser.add_argument('--config_file', default = 'config/voc320.py',
                        help = 'Configuration file.')
    parser.add_argument('--dataset_path', default='../VOCdevkit',
                        help = 'Path to the root directory of VOC dataset.')
    parser.add_argument('--batch_size', default = None, type = int,
                        help = 'If None, the value in the config_file is used instead.')
    parser.add_argument('--num_parallel_calls', default = 8, type = int,
                        help = 'Number of threads involved in loading and preprocessing data.')
    parser.add_argument('--prefetch_buffer', default = 100, type = int,
                        help = 'Maximum number of batches to have ready at a given time.')
    args = parser.parse_args()

    voc_evaluation(args.model_weights, args.config_file, args.dataset_path,
                    args.batch_size, args.num_parallel_calls, args.prefetch_buffer)
