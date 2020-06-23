import argparse
import numpy as np
import os
from os import path
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from models import RefineDetVGG16
from utils import read_jpeg_image, resize_image_and_boxes, absolute2relative
from voc import load_voc_dataset, Augmentation
from voc.config import IMAGE_SIZE, BATCH_SIZE, SHUFFLE_BUFFER, VOC_CLASSES, LR_SCHEDULE, MOMENTUM, NUM_EPOCHS, STEPS_PER_EPOCH


parser = argparse.ArgumentParser()
parser.add_argument('--voc_root', type=str, default='./data/VOCdevkit',
                    help='Path to the VOCdevkit directory.')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to the weights file, in the case of resuming training.')
parser.add_argument('--initial_epoch', type=int, default=0,
                    help='Starting epoch. Give a value bigger than zero to resume training.')
parser.add_argument('--batch_size', type=int, default=None,
                    help='Useful for quick tests. If not provided, the value in the config file is used instead.')
args = parser.parse_args()

BATCH_SIZE = args.batch_size or BATCH_SIZE


def build_dataset(img_paths, bboxes, repeat=False, shuffle=False,
                  drop_remainder=False, augmentation_fn=None):
    row_lengths = [len(img_bboxes) for img_bboxes in bboxes]
    bboxes_concat = np.concatenate(bboxes, axis=0)
    bboxes = tf.RaggedTensor.from_row_lengths(values=bboxes_concat,
                                              row_lengths=row_lengths)
    
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, bboxes))

    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(len(img_paths),
                                  reshuffle_each_iteration=True)

    dataset = dataset.map(lambda img_path, boxes:
                          (read_jpeg_image(img_path), boxes))

    if augmentation_fn:
        dataset = dataset.map(augmentation_fn)

    dataset = dataset.map(lambda image, boxes:
                          resize_image_and_boxes(image, boxes, IMAGE_SIZE))
    dataset = dataset.map(lambda image, boxes:
                          (image, absolute2relative(boxes, tf.shape(image))))

    # This hack is to allow batching into ragged tensors
    dataset = dataset.map(lambda image, boxes:
                          (image, tf.expand_dims(boxes, 0)))        
    dataset = dataset.map(lambda image, boxes:
                          (image, tf.RaggedTensor.from_tensor(boxes)))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=drop_remainder)
    dataset = dataset.map(lambda image, boxes:
                          (image, boxes.merge_dims(1, 2)))
    
    return dataset



train_img_paths, train_bboxes = load_voc_dataset(
    dataroot=args.voc_root,
    splits=[('VOC2007', 'trainval'), ('VOC2012', 'trainval')],
    keep_difficult = False, return_difficulty=False)
print('INFO: Loaded %d training samples' % len(train_img_paths))

test_img_paths, test_bboxes = load_voc_dataset(
    dataroot=args.voc_root,
    splits=[('VOC2007', 'test')],
    keep_difficult = True, return_difficulty=True)
print('INFO: Loaded %d testing samples' % len(test_img_paths))

train_data = build_dataset(train_img_paths, train_bboxes,
                           repeat=True, shuffle=True, drop_remainder=True,
                           augmentation_fn=Augmentation())
test_data = build_dataset(test_img_paths, test_bboxes,
                          repeat=False, shuffle=False, drop_remainder=False)

print(train_data)

print('INFO: Instantiating model...')

model = RefineDetVGG16(num_classes=len(VOC_CLASSES))
model.build(input_shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

if args.checkpoint:
    model.load_weights(args.checkpoint)
else:
    # If training from scratch, initialize only the base CNN
    # with pretrained Imagenet weights
    model.base.load_weights(
        path.join('weights', 'VGG_ILSVRC_16_layers_fc_reduced.h5'), by_name=True)


lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(*LR_SCHEDULE)
optimizer = tf.keras.optimizers.SGD(lr_scheduler, momentum=MOMENTUM)
optimizer.iterations = tf.Variable(STEPS_PER_EPOCH * args.initial_epoch)

print('Trainint at learning rate =', optimizer._decayed_lr(tf.float32))

model.compile(optimizer=optimizer)

os.makedirs('weights', exist_ok=True)
callbacks = [
    ModelCheckpoint(path.join('weights', 'refinedet_vgg16_{epoch:0d}.h5'),
                    monitor='total_loss')    
]

model.fit(train_data, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
          initial_epoch=args.initial_epoch, callbacks=callbacks)
