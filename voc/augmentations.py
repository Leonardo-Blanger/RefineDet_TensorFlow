import cv2
import numpy as np
import tensorflow as tf

import utils


def random_uniform(l=0.0, r=1.0):
    return tf.random.uniform(shape=(), minval=l, maxval=r)

def choice(prob):
    return random_uniform(0, 1) < prob


class RandomBrightness:
    def __init__(self, delta = 32, prob = 0.5):
        self.delta = delta
        self.prob = prob

    def __call__(self, image):
        def _f(image):
            return tf.image.random_brightness(image, self.delta)

        return tf.cond(choice(self.prob),
                       lambda: _f(image),
                       lambda: image)


class RandomContrast:
    def __init__(self, lower = 0.5, upper = 1.5, prob = 0.5):
        self.lower = lower
        self.upper = upper
        self.prob = prob

    def __call__(self, image):
        def _f(image):
            return tf.image.random_contrast(image, self.lower, self.upper)

        image = tf.cond(choice(self.prob),
                        lambda: _f(image),
                        lambda: image)
        return image


class RandomSaturation:
    def __init__(self, lower = 0.5, upper = 1.5, prob = 0.5):
        self.lower = lower
        self.upper = upper
        self.prob = prob

    def __call__(self, image):
        def _f(image):
            return tf.image.random_saturation(image, self.lower, self.upper)

        image = tf.cond(choice(self.prob),
                        lambda: _f(image),
                        lambda: image)
        return image


class RandomHue:
    def __init__(self, max_delta = 0.05, prob = 0.5):
        self.max_delta = max_delta
        self.prob = prob

    def __call__(self, image):
        def _f(image):
            return tf.image.random_hue(image, self.max_delta)
        
        image = tf.cond(choice(self.prob),
                        lambda: _f(image),
                        lambda: image)
        return image


class RandomShuffleChannels:
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, image):
        def _f(image):
            image = tf.transpose(image, [2, 0, 1])
            image = tf.random.shuffle(image)
            image = tf.transpose(image, [1, 2, 0])
            return image

        image = tf.cond(choice(self.prob),
                        lambda: _f(image),
                        lambda: image)
        return image


class PhotometricDistortions:
    def __init__(self):
        self.random_brightness = RandomBrightness(delta = 32, prob = 0.5)
        self.random_shuffle = RandomShuffleChannels(prob = 0.5)

        self.random_contrast = RandomContrast(lower = 0.5, upper = 1.5, prob = 0.5)
        self.random_saturation = RandomSaturation(lower = 0.5, upper = 1.5, prob = 0.5)
        self.random_hue = RandomHue(max_delta = 0.05, prob = 0.5)

        self.sequences = (
            [
                self.random_brightness,
                self.random_contrast,
                self.random_saturation,
                self.random_hue,
                self.random_shuffle
            ], [
                self.random_brightness,
                self.random_saturation,
                self.random_hue,
                self.random_contrast,
                self.random_shuffle
            ]
        )

    def __call__(self, image):
        def _f(i, image):
            for aug in self.sequences[i]:
                image = aug(image)
            return image

        image = tf.cond(choice(0.5),
                        lambda: _f(0, image),
                        lambda: _f(1, image))
        return image


class RandomExpansion:
    def __init__(self, max_ratio = 4, prob = 0.5, background = (123, 117, 104)):
        self.max_ratio = max_ratio
        self.prob = prob
        self.background = tf.constant(background, dtype=tf.float32)

    def __call__(self, image, boxes = None):
        def _f(image, boxes=None):
            height, width, channels = [tf.shape(image)[i] for i in range(3)]
            ratio = random_uniform(1.0, self.max_ratio)

            new_height = tf.cast(tf.cast(height, tf.float32) * ratio, tf.int32)
            new_width = tf.cast(tf.cast(width, tf.float32) * ratio, tf.int32)

            top = tf.random.uniform(shape=(), minval=0,
                                    maxval=new_height - height + 1, dtype=tf.int32)
            left = tf.random.uniform(shape=(), minval=0,
                                     maxval=new_width - width + 1, dtype=tf.int32)

            image = tf.image.pad_to_bounding_box(image - self.background, top, left,
                                                 new_height, new_width) + self.background

            if boxes is None:
                return image

            shift = tf.convert_to_tensor([left, top, left, top], dtype=boxes.dtype)
            boxes += tf.pad(shift, [[0, tf.shape(boxes)[1] - 4]])

            return image, boxes

        if boxes is None:
            return tf.cond(choice(self.prob),
                           lambda: _f(image, boxes),
                           lambda: image)
        else:
            return tf.cond(choice(self.prob),
                           lambda: _f(image, boxes),
                           lambda: (image, boxes))


class RandomCrop:
    def __init__(self,
                 min_ratio = 0.3,
                 min_aspect_ratio = 0.5,
                 max_aspect_ratio = 2.0,
                 prob = 5.0 / 6.0,
                 max_tries = 50):
         self.min_ratio = min_ratio
         self.min_aspect_ratio = min_aspect_ratio
         self.max_aspect_ratio = max_aspect_ratio
         self.prob = prob
         self.max_tries = max_tries

    def __call__(self, image, boxes = None):
        height, width = [tf.shape(image)[i] for i in range(2)]

        def _cond(try_cnt, image, boxes=None):
            return try_cnt < self.max_tries

        def _loop_body(try_cnt, image, boxes=None):
            ratio_height = random_uniform(self.min_ratio, 1.0)
            ratio_width = random_uniform(self.min_ratio, 1.0)

            crop_height = tf.cast(tf.cast(height, tf.float32) * ratio_height, tf.int32)
            crop_width = tf.cast(tf.cast(width, tf.float32) * ratio_width, tf.int32)

            top = tf.random.uniform(shape=(), minval=0,
                                    maxval=height - crop_height + 1, dtype=tf.int32)
            left = tf.random.uniform(shape=(), minval=0,
                                     maxval=width - crop_width + 1, dtype=tf.int32)
            
            def _adjust_boxes(try_cnt, image, boxes=None):
                if boxes is None:
                    image = tf.image.crop_to_bounding_box(image, top, left,
                                                          crop_height, crop_width)
                    return tf.constant(self.max_tries), image

                bottom = top + crop_height - 1
                right = left + crop_width - 1

                cx = (boxes[:, 0] + boxes[:, 2])*0.5
                cy = (boxes[:, 1] + boxes[:, 3])*0.5

                mask = tf.logical_and(
                    tf.logical_and(cx >= tf.cast(left, tf.float32),
                                   cx <= tf.cast(right, tf.float32)),
                    tf.logical_and(cy >= tf.cast(top, tf.float32),
                                   cy <= tf.cast(bottom, tf.float32))
                )

                def _resize(image, boxes):
                    image = tf.image.crop_to_bounding_box(image, top, left,
                                                          crop_height, crop_width)
                    boxes = tf.boolean_mask(boxes, mask=mask, axis=0)

                    shift = tf.convert_to_tensor([left, top, left, top], dtype=boxes.dtype)
                    boxes -= tf.pad(shift, [[0, tf.shape(boxes)[1] - 4]])
                    boxes = utils.clip_boxes(boxes, 0, 0, crop_width, crop_height)
                    
                    return tf.constant(self.max_tries), image, boxes

                return tf.cond(tf.reduce_any(mask),
                               lambda: _resize(image, boxes),
                               lambda: (try_cnt+1, image, boxes))


            ar = tf.cast(crop_height, tf.float32) / tf.cast(crop_width, tf.float32)
            check = tf.convert_to_tensor([ar >= self.min_aspect_ratio,
                                          ar <= self.max_aspect_ratio])

            if boxes is None:
                return tf.cond(tf.reduce_all(check),
                               lambda: _adjust_boxes(try_cnt, image),
                               lambda: (try_cnt+1, image))
            else:
                return tf.cond(tf.reduce_all(check),
                               lambda: _adjust_boxes(try_cnt, image, boxes),
                               lambda: (try_cnt+1, image, boxes))

        def _f(image, boxes=None):
            if boxes is None:
                _, image = tf.while_loop(_cond, _loop_body,
                                         [tf.constant(0), image])
                return image
            else:
                _, image, boxes = tf.while_loop(_cond, _loop_body,
                                                [tf.constant(0), image, boxes])
                return image, boxes

        if boxes is None:
            return tf.cond(choice(self.prob),
                           lambda: _f(image),
                           lambda: image)
        else:
            return tf.cond(choice(self.prob),
                           lambda: _f(image, boxes),
                           lambda: (image, boxes))


class RandomHorizontalFlip:
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, image, boxes = None):
        def _f(image, boxes=None):
            image = tf.image.flip_left_right(image)

            if boxes is None:
                return image
            
            width = tf.cast(tf.shape(image)[1], dtype=boxes.dtype)
            xmin, ymin, xmax, ymax = [boxes[:,i] for i in range(4)]
            xmin, xmax = width - xmax, width - xmin
            stacked_boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
            boxes = tf.concat([stacked_boxes, boxes[:, 4:]], axis=-1)

            return image, boxes

            
        return tf.cond(choice(self.prob),
                       lambda: _f(image, boxes),
                       lambda: image if boxes is None else (image, boxes))


class Augmentation:
    def __init__(self, background = (123, 117, 104)):
        self.photometric_distortions = PhotometricDistortions()

        self.geometric_augmentations = [
            RandomExpansion(max_ratio = 4,
                            prob = 0.5,
                            background = background),

            RandomCrop(min_ratio = 0.3,
                       min_aspect_ratio = 0.5,
                       max_aspect_ratio = 2.0,
                       prob = 5.0 / 6.0,
                       max_tries = 50),

            RandomHorizontalFlip(prob = 0.5)
        ]

    def __call__(self, image, boxes = None):
        image = tf.cast(image, tf.float32)
        if boxes is not None:
            boxes = tf.cast(boxes, tf.float32)
        image = self.photometric_distortions(image)

        for aug in self.geometric_augmentations:
            if boxes is None:
                image = aug(image)
            else:
                image, boxes = aug(image, boxes)

        return image, boxes
