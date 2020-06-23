import argparse
import numpy as np
from os import path
import tensorflow as tf
from tqdm import tqdm

from metrics import MeanAveragePrecision
from models import RefineDetVGG16
from utils import read_jpeg_image, resize_image_and_boxes, absolute2relative
from voc import load_voc_dataset
from voc.config import IMAGE_SIZE, BATCH_SIZE, SHUFFLE_BUFFER, VOC_CLASSES, LR_SCHEDULE, MOMENTUM


parser = argparse.ArgumentParser()
parser.add_argument('--voc_root', type=str, default='./data/VOCdevkit',
                    help='Path to the VOCdevkit directory.')
parser.add_argument('--final_checkpoint', type=str, default='weights/refinedet_vgg16_24.h5',
                    help='Path to the trained weights file.')
parser.add_argument('--batch_size', type=int, default=None,
                    help='Useful for quick tests. If not provided, the value in the config file is used instead.')
args = parser.parse_args()

BATCH_SIZE = args.batch_size or BATCH_SIZE


def build_dataset(img_paths, bboxes, repeat=False,
                  shuffle=False, drop_remainder=False):
    row_lengths = [len(img_bboxes) for img_bboxes in bboxes]
    bboxes_concat = np.concatenate(bboxes, axis=0)
    bboxes = tf.RaggedTensor.from_row_lengths(values=bboxes_concat,
                                              row_lengths=row_lengths)

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, bboxes))
    
    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(len(img_paths))

    dataset = dataset.map(lambda img_path, boxes:
                          (read_jpeg_image(img_path), boxes))
    dataset = dataset.map(lambda image, boxes:
                          resize_image_and_boxes(image, boxes, IMAGE_SIZE))
    dataset = dataset.map(lambda image, boxes:
                          (image, absolute2relative(boxes, list(image.shape))))

    # This hack is to allow batching into ragged tensors
    dataset = dataset.map(lambda image, boxes:
                          (image, tf.expand_dims(boxes, 0)))        
    dataset = dataset.map(lambda image, boxes:
                          (image, tf.RaggedTensor.from_tensor(boxes)))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=drop_remainder)
    dataset = dataset.map(lambda image, boxes:
                          (image, boxes.merge_dims(1, 2)))
    
    dataset = dataset.prefetch(20)
    return dataset


test_img_paths, test_bboxes = load_voc_dataset(
    dataroot=args.voc_root,
    splits=[('VOC2007', 'test')],
    keep_difficult = True, return_difficulty = True)
print('INFO: Loaded %d testing samples' % len(test_img_paths))

test_data = build_dataset(test_img_paths, test_bboxes,
                          repeat=False, shuffle=False, drop_remainder=False)

print('INFO: Instantiating model...')

model = RefineDetVGG16(num_classes=len(VOC_CLASSES))
model.build(input_shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
model.load_weights(args.final_checkpoint)

y_true = []
y_pred = []
difficulty = []

for x, yt in tqdm(test_data, desc='Running predictions'):
    yp = model(x, decode=True)
    y_true += [y[:,:-1] for y in yt]
    y_pred += [y for y in yp]
    difficulty += [y[:,-1] for y in yt]


meanAP_metric = MeanAveragePrecision()
meanAP_metric.update_state(y_true, y_pred)
meanAP_unfiltered = meanAP_metric.per_class_AP()

meanAP_metric.reset_state()
meanAP_metric.update_state(y_true, y_pred, ignore_samples=difficulty)
meanAP_filtered = meanAP_metric.per_class_AP()

for cls, mAP_uf, mAP_f in zip(VOC_CLASSES[1:], meanAP_unfiltered, meanAP_filtered):
    print('Average Precision for class \'{}\':'.format(cls), end="")
    print(' {:.2f}% (unfiltered) --'.format(100*mAP_uf), end="")
    print(' {:.2f}% (difficult filtered)'.format(100*mAP_f))

print('\nFinal Mean Average Precision:', end="")
print(' {:.2f}% (unfiltered) --'.format(100*np.mean(meanAP_unfiltered)), end="")
print(' {:.2f}% (difficult filtered)'.format(100*np.mean(meanAP_filtered)))
