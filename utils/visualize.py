import cv2 as cv
import imgaug as ia
import copy
import os

def maybe_load(image):
    if isinstance(image, str):
        image_file = image
        if not os.path.exists(image_file):
            raise Exception('Image not found: %s' % image_file)
        else:
            image = cv.imread(image_file)
            if image is None:
                raise Exception('Error loading image: %s' % image_file)
    return image

def display_image(image, bnd_boxes = None, channels = 'bgr'):
    image = maybe_load(image)

    channels = channels.lower()
    if channels == 'bgr':
        pass
    elif channels == 'rgb':
        image = image[..., [2,1,0]]
    else:
        raise Exception('Channels format not supported: %s' % channels)

    if not bnd_boxes is None:
        bnd_boxes = ia.BoundingBoxesOnImage(
            [ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label)
            for x1, y1, x2, y2, label in bnd_boxes],
            shape = image.shape
        )
        image = bnd_boxes.draw_on_image(image, color = (0,255,0), thickness = 3)

    cv.imshow('Image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
