import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ZeroPadding2D

from .custom_layers import L2Norm
from .modules import DetectionBlock, TCBBlock
from .refinedet_base import RefineDetBase
from .vgg16_reducedfc import VGG16ReducedFC


class RefineDetVGG16(RefineDetBase):
    def __init__(self, num_classes, aspect_ratios=[0.5, 1.0, 2.0],
                 scales=[0.1, 0.2, 0.4, 0.8], **kwargs):
        self.num_classes = num_classes
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.num_scales = 4
        
        super(RefineDetVGG16, self).__init__(**kwargs)

        if len(self.scales) != self.num_scales:
            raise Exception('Wrong number of scales provided: len({}) != {}'.format(
                self.scales, self.num_scales))

        l2_reg = tf.keras.regularizers.l2(5e-4)

        self.base = VGG16ReducedFC(output_features=['conv4_3', 'conv5_3', 'fc7'],
                                   kernel_regularizer=l2_reg)

        self.feat4_3_norm = L2Norm(10.0)
        self.feat5_3_norm = L2Norm(8.0)

        self.extras = tf.keras.Sequential([
            Conv2D(256, kernel_size=1, strides=1, padding='same', activation='relu',
                   kernel_regularizer=l2_reg, name='extra_conv1'),
            ZeroPadding2D(1),
            Conv2D(512, kernel_size=3, strides=2, padding='valid', activation='relu',
                   kernel_regularizer=l2_reg, name='extra_conv2')
        ])

        self.arm_blocks = []
        self.odm_blocks = []
        self.tcb_blocks = []
        
        for i, scale in enumerate(self.scales):
            arm_block = DetectionBlock(num_classes=2,
                                       scale=scale,
                                       aspect_ratios=self.aspect_ratios,
                                       name='arm_block%d'%(i+1),
                                       kernel_regularizer=l2_reg)
            odm_block = DetectionBlock(num_classes=self.num_classes,
                                       scale=scale,
                                       aspect_ratios=self.aspect_ratios,
                                       name='odm_block%d'%(i+1),
                                       kernel_regularizer=l2_reg)
            tcb_block = TCBBlock(name='tcb_block%d'%(i+1),
                                 kernel_regularizer=l2_reg)
            self.arm_blocks.append(arm_block)
            self.odm_blocks.append(odm_block)
            self.tcb_blocks.append(tcb_block)
            

    @tf.function
    def call(self, x, training=False, decode=False):
        ftm4_3, ftm5_3, ftm7 = self.base(x)
        ftm4_3_norm = self.feat4_3_norm(ftm4_3)
        ftm5_3_norm = self.feat5_3_norm(ftm5_3)
        ext = self.extras(ftm7)
        
        ftmaps = [ftm4_3_norm, ftm5_3_norm, ftm7, ext]
        
        arm_cls, arm_loc, anchors = [], [], []
        for i, ftmap in enumerate(ftmaps):
            cls, loc, anc = self.arm_blocks[i](ftmap, return_anchors=True)
            arm_cls.append(cls)
            arm_loc.append(loc)
            anchors.append(anc)

        tcb_ftmaps = []
        next_ftm = None
        for i in range(self.num_scales-1, -1, -1):
            cur_ftm = self.tcb_blocks[i](ftmaps[i], next_ftm)
            tcb_ftmaps.append(cur_ftm)
            next_ftm = cur_ftm
        tcb_ftmaps.reverse()

        odm_cls, odm_loc = [], []
        for i, ftmap in enumerate(tcb_ftmaps):
            cls, loc = self.odm_blocks[i](ftmap)
            odm_cls.append(cls)
            odm_loc.append(loc)

        arm_cls = tf.concat(arm_cls, axis=1)
        arm_loc = tf.concat(arm_loc, axis=1)
        odm_cls = tf.concat(odm_cls, axis=1)
        odm_loc = tf.concat(odm_loc, axis=1)
        anchors = tf.concat(anchors, axis=0)

        output = (arm_cls, arm_loc), (odm_cls, odm_loc), anchors

        if decode and not training:
            output = self.decode(output)

        return output
