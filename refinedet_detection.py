import argparse
import importlib
import os
from tqdm import tqdm
import cv2
import numpy as np

from models import RefineDetVGG16

def detect(backbone = 'VGG16',
           model_weights = 'weights/RefineDetVOC_320.h5',
           config = 'config/voc320.py',
           images = 'samples/',
           output_dir = 'samples/detections/',
           display = False):

   if isinstance(config, str):
       # Get the configuration module
       config_module = '.'.join(config.split('/'))[:-3]
       config = importlib.import_module(config_module)

   if not os.path.isdir(output_dir):
       os.makedirs(output_dir)

   print('Building the model...')

   if backbone == 'VGG16':
       model = RefineDetVGG16(num_classes = len(config.classes),
                              background_id = config.classes.index('background'))
   else:
       raise Exception('Unknown backbone:', backbone)

   model.build(input_shape = (None, config.image_size, config.image_size, 3))
   model.summary()

   print('Loading weights...')
   model.load_weights(model_weights)

   if os.path.isdir(images):
       images = [os.path.join(images, file) for file in os.listdir(images)]
   else:
       images = [images]

   for image_file in tqdm(images, desc='Running the model on images...'):
       image = cv2.imread(image_file)

       if image is None:
           print('Bad file:', image_file, ' - Skipping...')
           continue

       input_img = cv2.resize(image, (config.image_size, config.image_size))
       input_img = (input_img - config.channel_means).astype('float32')
       input_img = np.expand_dims(input_img, axis = 0)

       boxes = model(input_img, decode = True)[0].numpy()

       for box in boxes:
           box[0] *= image.shape[1] / config.image_size
           box[1] *= image.shape[0] / config.image_size
           box[2] *= image.shape[1] / config.image_size
           box[3] *= image.shape[0] / config.image_size
           box = box.astype('int32')
           image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)

       det_filename = '.'.join(image_file.split('/')[-1].split('.')[:-1]) + '_det.png'
       cv2.imwrite(os.path.join(output_dir, det_filename), image)

       if display:
           cv2.imshow(det_filename, image)
           cv2.waitKey(0)
           cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        '=== RefineDet Object Detector ==='
    )
    parser.add_argument('--backbone', default = 'VGG16',
                        help = 'Backbone CNN to use. Currently, only VGG16 is supported.')

    parser.add_argument('--model_weights', default = 'weights/RefineDetVOC_320.h5',
                        help = 'Path to the weights file.')

    parser.add_argument('--config_file', default = 'config/voc320.py',
                        help = 'Path to the model configuration file.')

    parser.add_argument('--images', default = 'samples/',
                        help = 'Either the path to an image, or to a directory containing the images.')

    parser.add_argument('--output_dir', default = 'samples/detections/',
                        help = 'Where to save the images with the drawn detections.')

    parser.add_argument('--display', default = 0, type = int,
                        help = 'Set this flag if you want to see the detections as they happen.')

    args = parser.parse_args()

    detect(args.backbone, args.model_weights, args.config_file,
                                args.images, args.output_dir, args.display)
