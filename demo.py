import imgaug as ia
import matplotlib.pyplot as plt
import numpy as np
import os
from os import path
from PIL import Image

from models import RefineDetVGG16
from voc.config import VOC_CLASSES, IMAGE_SIZE

MODEL_FILE = path.join('weights', 'refinedet_vgg16_24.h5')
IMAGES_DIR = 'samples'
OUTPUT_DIR = path.join(IMAGES_DIR, 'detections')
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = RefineDetVGG16(num_classes=len(VOC_CLASSES), conf_threshold=0.5)
model.build((None,) + IMAGE_SIZE + (3,))
model.load_weights(MODEL_FILE)

for image_file in os.listdir(IMAGES_DIR):
    try:
        image = Image.open(path.join(IMAGES_DIR, image_file))
    except IsADirectoryError:
        continue

    
    image_inp = image.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    image_inp = np.array(image_inp).astype('float32')
    image_inp = np.expand_dims(image_inp, 0)
    
    bboxes = model(image_inp, decode=True)[0]
    
    bboxes = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=VOC_CLASSES[int(label)])
        for x1, y1, x2, y2, label, conf in bboxes
    ], shape=(1,1))

    image = np.array(image)
    bboxes = bboxes.on(image)
    image = bboxes.draw_on_image(image, color=[0,0,255], size=2)

    output_file = '.'.join(image_file.split('.')[:-1]) + '_det.jpg'
    Image.fromarray(image).save(path.join(OUTPUT_DIR, output_file))
    plt.imshow(image)
    plt.show()
