import tensorflow as tf
tf.enable_eager_execution()

import argparse
import importlib
import numpy as np
import os
import cv2
import time

from parsers import voc_parser
from utils import visualize
from metrics import refinedet_losses
from augmentations import SSDAugmentation

from models.vgg16_reducedfc import VGG16ReducedFC
from models import RefineDetVGG16

parser = argparse.ArgumentParser(
    description = 'RefineDet detector training for PASCAL VOC 07+12'
)

parser.add_argument('--dataset_path', default='../VOCdevkit',
                    help = 'Path to the root directory of VOC dataset.')

parser.add_argument('--config_file', default = 'config/voc320.py',
                    help = 'Configuration file.')

parser.add_argument('--batch_size', default = None, type = int,
                    help = 'If None, the value in the config_file is used instead.')

parser.add_argument('--first_iteration', default = 0, type = int,
                    help = 'First iteration (0 based)')

parser.add_argument('--num_iterations', default = None, type = int,
                    help = 'If None, the value in the config file is used instead.')

parser.add_argument('--display_loss_frequency', default = 10, type = int,
                    help = 'Number of iterations between two loss printings.')

parser.add_argument('--save_weights_frequency', default = 1000, type = int,
                    help = 'Number of iterations between each model checkpoint.')

parser.add_argument('--weights_dir', default = 'weights', type = str,
                    help = 'directory to save the model weight checkpoints.')

parser.add_argument('--load_weights', default = None, type = str,
                    help = 'Path to a saved weights file. Use this to resume training.')

parser.add_argument('--num_parallel_calls', default = 8, type = int,
                    help = 'Number of threads involved in loading and preprocessing data.')

parser.add_argument('--shuffle_buffer_size', default = 1000, type = int,
                    help = 'Size of buffer used for data shuffling.')

parser.add_argument('--prefetch_buffer', default = 100, type = int,
                    help = 'Maximum number of batches to have ready at a given time.')

args = parser.parse_args()

dataset_path = args.dataset_path
config_file = args.config_file
batch_size = args.batch_size
first_iteration = args.first_iteration
num_iterations = args.num_iterations
display_loss_frequency = args.display_loss_frequency
save_weights_frequency = args.save_weights_frequency
weights_dir = args.weights_dir
load_weights = args.load_weights
num_parallel_calls = args.num_parallel_calls
shuffle_buffer_size = args.shuffle_buffer_size
prefetch_buffer = args.prefetch_buffer


# Get the configuration module
config_module = '.'.join(config_file.split('/'))[:-3]
config = importlib.import_module(config_module)

image_size = config.image_size
channel_means = config.channel_means
pos_iou_threshold = config.pos_iou_threshold
neg_iou_threshold = config.neg_iou_threshold
anchor_refinement_threshold = config.anchor_refinement_threshold
variances = config.variances
learning_rates = config.learning_rates
momentum = config.momentum
weight_decay = config.weight_decay
num_classes = len(config.voc_classes)

if batch_size is None: batch_size = config.batch_size
if num_iterations is None: num_iterations = config.num_iterations

model = RefineDetVGG16()
model.build(input_shape = (None, image_size, image_size, 3))
anchor_boxes = model.build_anchors(input_shape = (image_size, image_size, 3))

if load_weights is None:
    model.base_model.load_weights(os.path.join(weights_dir, 'vgg16_reducedfc.h5'))
else:
    model.load_weights(load_weights)

model.summary()

# Obtain the data
def get_data_paths(image_sets = [('2007', 'trainval'), ('2012', 'trainval')]):
    image_paths, annotation_paths = [], []

    for year, image_set in image_sets:
        ids_file = os.path.join(
            dataset_path, 'VOC'+year, 'ImageSets', 'Main', image_set+'.txt')

        with open(ids_file, 'r') as f:
            ids = [line.strip() for line in f]

        image_paths += [
            os.path.join(dataset_path, 'VOC'+year, 'JPEGImages', id+'.jpg')
            for id in ids
        ]
        annotation_paths += [
            os.path.join(dataset_path, 'VOC'+year, 'Annotations', id+'.xml')
            for id in ids
        ]

    return image_paths, annotation_paths

train_images, train_annotations = get_data_paths([('2007', 'trainval'), ('2012', 'trainval')])
test_images, test_annotations = get_data_paths([('2007', 'test')])

print('%d train images' % len(train_images))
print('%d test images' % len(test_images))

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_annotations))

train_dataset = train_dataset.shuffle(
    buffer_size = shuffle_buffer_size,
    reshuffle_each_iteration = True
)

# Augmentation policy
augmentation = SSDAugmentation()

# Method for loading and preprocessing a single data sample
def load_and_preprocess(img_path, ann_path):
    img_path = img_path.numpy().decode("utf-8")
    ann_path = ann_path.numpy().decode("utf-8")

    image = cv2.imread(img_path)
    boxes = voc_parser.parse(ann_path, classes = config.voc_classes)

    #image = image.astype('float32')
    image, boxes = augmentation(image, boxes)

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

train_dataset = train_dataset.map(
    lambda image, boxes: tf.py_function(
        load_and_preprocess, [image, boxes], [tf.float32, tf.float32]
    ),
    num_parallel_calls = num_parallel_calls
)

train_dataset = train_dataset.padded_batch(batch_size,
    padded_shapes = ([image_size, image_size, 3], [None, 5]),
    drop_remainder = True
)

train_dataset = train_dataset.prefetch(prefetch_buffer)
batch_iterator = iter(train_dataset)


# Weight decay
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        for layer in layer.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) or \
                    isinstance(layer, tf.keras.layers.Conv2DTranspose):
                layer.kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    else:
        if isinstance(layer, tf.keras.layers.Conv2D) or \
                isinstance(layer, tf.keras.layers.Conv2DTranspose):
            layer.kernel_regularizer = tf.keras.regularizers.l2(weight_decay)


learning_rate = None
next_lr_update = None

for it, lr in learning_rates:
    if it <= first_iteration:
        learning_rate = lr
    else:
        next_lr_update = it
        break

global_step = tf.Variable(first_iteration, trainable=False)

def learning_rate_scheduler():
    global global_step, next_lr_update, learning_rate

    if global_step.numpy() == next_lr_update:
        for it, lr in learning_rates:
            if it <= next_lr_update:
                learning_rate = lr
            else:
                next_lr_update = it
                break
    return learning_rate

optimizer = tf.train.MomentumOptimizer(
    learning_rate = learning_rate_scheduler,
    momentum = momentum
)

acum_arm_cls_loss = 0
acum_arm_loc_loss = 0
acum_odm_cls_loss = 0
acum_odm_loc_loss = 0
acum_time = 0

for iteration in range(first_iteration, num_iterations+1):
    start_time = time.time()

    try:
        images, batch_boxes = next(batch_iterator)
    except:
        batch_iterator = iter(train_dataset)
        images, batch_boxes = next(batch_iterator)

    batch_boxes = [
        tf.convert_to_tensor([box for box in boxes if box[4].numpy() > 0])
        for boxes in batch_boxes
    ]

    with tf.GradientTape() as tape:
        output = model(images, decode = False)

        arm_cls_loss, arm_loc_loss, odm_cls_loss, odm_loc_loss = refinedet_losses(
            ground_truth = batch_boxes,
            output = output,
            anchors = anchor_boxes,
            num_classes = num_classes,
            background_id = 0,
            pos_iou_threshold = pos_iou_threshold,
            neg_iou_threshold = neg_iou_threshold,
            variances = variances
        )

        total_loss = arm_cls_loss + arm_loc_loss + odm_cls_loss + odm_loc_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)
    end_time = time.time()

    acum_arm_cls_loss += arm_cls_loss
    acum_arm_loc_loss += arm_loc_loss
    acum_odm_cls_loss += odm_cls_loss
    acum_odm_loc_loss += odm_loc_loss
    acum_time += end_time - start_time

    if iteration and iteration % display_loss_frequency == 0:
        print('Iteration %d losses:' % iteration)
        print('arm_cls: %.3f arm_loc: %.3f odm_cls: %.3f odm_loc: %.3f' % (
            acum_arm_cls_loss / display_loss_frequency,
            acum_arm_loc_loss / display_loss_frequency,
            acum_odm_cls_loss / display_loss_frequency,
            acum_odm_loc_loss / display_loss_frequency
        ))
        print('Avg Iteration Time: %.2f sec' % (acum_time / display_loss_frequency))
        print()

        acum_arm_cls_loss = 0
        acum_arm_loc_loss = 0
        acum_odm_cls_loss = 0
        acum_odm_loc_loss = 0
        acum_time = 0

    if iteration and iteration % save_weights_frequency == 0:
        model.save_weights(os.path.join(weights_dir,
            'RefineDetVOC_%d_%d.h5' % (image_size, iteration)
        ))
