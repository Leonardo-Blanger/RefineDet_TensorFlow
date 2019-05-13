import numpy as np
import cv2

def choice(prob):
    return np.random.uniform(0, 1) < prob

def IOU(box0, boxes):
    xmin1, ymin1, xmax1, ymax1 = box0[:4]
    xmin2, ymin2, xmax2, ymax2 = [boxes[:,i] for i in range(4)]

    inter_width  = np.maximum(0, np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2))
    inter_height = np.maximum(0, np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2))

    inter_area = inter_width * inter_height
    union_area = (xmax1-xmin1)*(ymax1-ymin1) + (xmax2-xmin2)*(ymax2-ymin2) - inter_area
    return inter_area / union_area

class RandomBrightness:
    def __init__(self, delta = 32, prob = 0.5):
        self.delta = delta
        self.prob = prob

    def __call__(self, image):
        if choice(self.prob):
            image += np.random.uniform(-self.delta, self.delta)
        return image

class RandomShuffleChannels:
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, image):
        if choice(self.prob):
            image = image[..., np.random.permutation(3)]
        return image

class RandomContrast:
    def __init__(self, lower = 0.5, upper = 1.5, prob = 0.5):
        self.lower = lower
        self.upper = upper
        self.prob = prob

    def __call__(self, image):
        if choice(self.prob):
            factor = np.random.uniform(self.lower, self.upper)
            image = (image - 127.5) * factor + 127.5
            #image *= factor
        return image

class ConvertColors:
    def __init__(self, original = 'BGR', to = 'HSV'):
        self.original = original
        self.to = to

    def __call__(self, image):
        if self.original == 'BGR' and self.to == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.original == 'HSV' and self.to == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image

class RandomSaturation:
    def __init__(self, lower = 0.5, upper = 1.5, prob = 0.5):
        self.lower = lower
        self.upper = upper
        self.prob = prob

    def __call__(self, image):
        if choice(self.prob):
            image[:,:,1] *= np.random.uniform(self.lower, self.upper)
        return image

class RandomHue:
    def __init__(self, max_delta = 18, prob = 0.5):
        self.max_delta = max_delta
        self.prob = prob

    def __call__(self, image):
        if np.random.uniform(0, 1) < self.prob:
            image[:,:,0] += np.random.uniform(-self.max_delta, self.max_delta)
            image[:,:,0] += 360 * (image[:,:,0] < 0) - 360 * (image[:,:,0] > 360)
        return image

class PhotometricDistortions:
    def __init__(self):
        self.random_brightness = RandomBrightness(delta = 32, prob = 0.5)
        self.random_shuffle = RandomShuffleChannels(prob = 0.5)

        self.random_contrast = RandomContrast(lower = 0.5, upper = 1.5, prob = 0.5)
        self.random_saturation = RandomSaturation(lower = 0.5, upper = 1.5, prob = 0.5)
        self.random_hue = RandomHue(max_delta = 18, prob = 0.5)

        self.bgr2hsv = ConvertColors(original = 'BGR', to = 'HSV')
        self.hsv2bgr = ConvertColors(original = 'HSV', to = 'BGR')

        self.sequences = [
            [
                self.random_brightness,

                self.random_contrast,
                self.bgr2hsv,
                self.random_saturation,
                self.random_hue,
                self.hsv2bgr,

                self.random_shuffle
            ],
            [
                self.random_brightness,

                self.bgr2hsv,
                self.random_saturation,
                self.random_hue,
                self.hsv2bgr,
                self.random_contrast,

                self.random_shuffle
            ]
        ]

    def __call__(self, image, boxes = None):
        sequence = self.sequences[np.random.randint(2)]

        for aug in sequence:
            image = aug(image)

        if boxes is None:
            return image
        else:
            return image, boxes

class RandomExpansion:
    def __init__(self, max_ratio = 4, prob = 0.5, background = (104, 117, 123)):
        self.max_ratio = max_ratio
        self.prob = prob
        self.background = background

    def __call__(self, image, boxes = None):
        if choice(self.prob):
            ratio = np.random.uniform(1, self.max_ratio)

            img_height, img_width = image.shape[:2]
            new_height = int(img_height * ratio)
            new_width = int(img_width * ratio)

            top = int(np.random.uniform(0, new_height - img_height))
            left = int(np.random.uniform(0, new_width - img_width))

            new_image = np.zeros((new_height, new_width, 3)) + self.background
            new_image[top:top+img_height, left:left+img_width] = image
            image = new_image

            if boxes is not None:
                boxes[:, (0, 2)] += left
                boxes[:, (1, 3)] += top

        if boxes is None:
            return image
        else:
            return image, boxes

class RandomCrop:
    def __init__(self,
                 min_ratio = 0.5,
                 min_aspect_ratio = 0.5,
                 max_aspect_ratio = 2.0,
                 min_box_ious = [None, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
                 max_tries = 50):
         self.min_ratio = min_ratio
         self.min_aspect_ratio = min_aspect_ratio
         self.max_aspect_ratio = max_aspect_ratio
         self.min_box_ious = min_box_ious
         self.max_tries = max_tries

    def __call__(self, image, boxes = None):
        img_height, img_width = image.shape[:2]

        while True:
            min_iou = np.random.choice(self.min_box_ious)

            if min_iou is None:
                if boxes is None:
                    return image
                else:
                    return image, boxes

            for _ in range(self.max_tries):
                ratio_height = np.random.uniform(self.min_ratio, 1)
                ratio_width = np.random.uniform(self.min_ratio, 1)

                crop_height = int(ratio_height * img_height)
                crop_width = int(ratio_width * img_width)

                ar = float(crop_height) / crop_width
                if ar < self.min_aspect_ratio or ar > self.max_aspect_ratio:
                    continue

                top = int(np.random.uniform(img_height - crop_height))
                left = int(np.random.uniform(img_width - crop_width))

                if boxes is not None:
                    crop_box = [left, top, left+crop_width, top+crop_height]
                    box_ious = IOU(crop_box, boxes)

                    if np.min(box_ious) < min_iou:
                        continue

                    boxes_xc = (boxes[:, 0] + boxes[:, 2]) * 0.5
                    boxes_yc = (boxes[:, 1] + boxes[:, 3]) * 0.5

                    mask1 = boxes_xc >= crop_box[0]
                    mask2 = boxes_xc <= crop_box[2]
                    mask3 = boxes_yc >= crop_box[1]
                    mask4 = boxes_yc <= crop_box[3]
                    mask = mask1 * mask2 * mask3 * mask4

                    if not mask.any():
                        continue

                    boxes = boxes[mask]

                    boxes[:, (0, 2)] -= left
                    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, crop_width)

                    boxes[:, (1, 3)] -= top
                    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, crop_height)

                image = image[top:top+crop_height, left:left+crop_width]

                if boxes is None:
                    return image
                else:
                    return image, boxes

class RandomHorizontalFlip:
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, image, boxes = None):
        if choice(self.prob):
            image = image[:, ::-1]
            if boxes is not None:
                boxes[:, (0, 2)] = image.shape[1] - boxes[:, (2, 0)]
        return image, boxes


class SSDAugmentation:
    def __init__(self, background = (104, 117, 123), channel_order = 'BGR'):
        self.channel_order = channel_order

        self.photometric_distortions = PhotometricDistortions()

        self.random_expansion = RandomExpansion(max_ratio = 4,
                                                prob = 0.5,
                                                background = background)

        self.random_crop = RandomCrop(min_ratio = 0.3,
                                      min_aspect_ratio = 0.5,
                                      max_aspect_ratio = 2.0,
                                      min_box_ious = [None, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
                                      max_tries = 50)

        self.random_horizontal_flip = RandomHorizontalFlip(prob = 0.5)

    def __call__(self, image, boxes = None):
        if self.channel_order == 'RGB':
            image = image[..., (2,1,0)]
        image = image.astype('float32')

        image = self.photometric_distortions(image)

        if boxes is None:
            image = self.random_expansion(image)
            image = self.random_crop(image)
            image = self.random_horizontal_flip(image)
            return image
        else:
            image, boxes = self.random_expansion(image, boxes)
            image, boxes = self.random_crop(image, boxes)
            image, boxes = self.random_horizontal_flip(image, boxes)
            return image, boxes
