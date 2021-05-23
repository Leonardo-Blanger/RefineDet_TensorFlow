import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D


class DetectionBlock(tf.keras.Model):
    def __init__(self, num_classes, aspect_ratios=[0.5, 1.0, 2.0], scale=1.0,
                 kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                 **kwargs):
        super(DetectionBlock, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.aspect_ratios = aspect_ratios
        self.boxes_per_cell = len(self.aspect_ratios)
        self.scale = scale

        self.conv_cls = Conv2D(num_classes * self.boxes_per_cell, kernel_size=3,
                               padding='same', activation=None,
                               kernel_regularizer=kernel_regularizer)
        self.conv_loc = Conv2D(4 * self.boxes_per_cell, kernel_size=3,
                               padding='same', activation=None,
                               kernel_regularizer=kernel_regularizer)


    def call(self, x, return_anchors=False):
        cls = self.conv_cls(x)
        loc = self.conv_loc(x)

        ftmap_height = x.get_shape().as_list()[1]
        ftmap_width = x.get_shape().as_list()[2]
        num_cells = ftmap_height * ftmap_width * self.boxes_per_cell

        cls = tf.reshape(cls, [-1, num_cells, self.num_classes])
        loc = tf.reshape(loc, [-1, num_cells, 4])
        cls = tf.nn.softmax(cls, axis=-1)

        if return_anchors:
            anchors = self.build_anchors(ftmap_height, ftmap_width)
            return cls, loc, anchors
        
        return cls, loc


    @tf.function
    def build_anchors(self, feature_height, feature_width):
        cy, cx = np.meshgrid(range(feature_height), range(feature_width),
                             indexing = 'ij')
        cx = (cx.astype('float32') + 0.5) / feature_width
        cy = (cy.astype('float32') + 0.5) / feature_height

        cx = np.expand_dims(cx, axis=-1)
        cy = np.expand_dims(cy, axis=-1)
        cx = np.repeat(cx, repeats=self.boxes_per_cell, axis=-1)
        cy = np.repeat(cy, repeats=self.boxes_per_cell, axis=-1)

        width = np.zeros_like(cx)
        height = np.zeros_like(cy)

        for i in range(self.boxes_per_cell):
            width[..., i] = min(1.0, self.scale * np.sqrt(self.aspect_ratios[i]))
            height[..., i] = min(1.0, self.scale / np.sqrt(self.aspect_ratios[i]))

        xmin = cx - width * 0.5
        ymin = cy - height * 0.5
        xmax = cx + width * 0.5
        ymax = cy + height * 0.5
        anchors = np.stack([xmin, ymin, xmax, ymax], axis=-1)
        
        return anchors.reshape((-1, 4))
