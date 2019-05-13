import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, Reshape, Concatenate, ZeroPadding2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
import numpy as np

from .vgg16_reducedfc import VGG16ReducedFC
from .layers import L2Norm
import output_encoder
from metrics import IOU

class RefineDetVGG16(tf.keras.Model):
    def __init__(self,
                 num_classes = 21,
                 background_id = 0,
                 aspect_ratios = [0.5, 1.0, 2.0],
                 scales = [0.1, 0.2, 0.4, 0.8],
                 variances = [0.1, 0.1, 0.2, 0.2],
                 conf_threshold = 0.1,
                 max_preds_per_class = 5,
                 nms_threshold = 0.45,
                 anchor_refinement_threshold = 0.99):
        super(RefineDetVGG16, self).__init__()

        self.num_classes = num_classes
        self.background_id = background_id
        self.boxes_per_cell = len(aspect_ratios)
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.variances = np.array(variances)
        self.conf_threshold = conf_threshold
        self.max_preds_per_class = max_preds_per_class
        self.nms_threshold = nms_threshold
        self.anchor_refinement_threshold = anchor_refinement_threshold

        if len(scales) != 4:
            raise Exception('scales must have length exactly 4.')

        self.num_output_features = 4

        self.base_model = VGG16ReducedFC(
            features_to_return = ['conv4_3', 'conv5_3', 'conv7']
        )

        self.conv4_3_norm = L2Norm(10)
        self.conv5_3_norm = L2Norm(8)

        self.extra_conv1 = Conv2D(256, kernel_size = (1, 1), padding = 'same', activation = 'relu', name = 'extra_conv1')
        self.extra_conv2 = Conv2D(512, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', activation = 'relu', name = 'extra_conv2')

        self.arm_cls = []
        self.arm_loc = []

        self.odm_cls = []
        self.odm_loc = []

        self.tcb_block1 = []
        self.tcb_block2 = []
        self.tcb_upsample = []

        for i in range(self.num_output_features):
            self.arm_cls.append(
                Conv2D(self.boxes_per_cell * 2,
                       kernel_size = (3, 3),
                       padding = 'same',
                       name = 'arm_cls_%d'%i)
            )
            self.arm_loc.append(
                Conv2D(self.boxes_per_cell * 4,
                       kernel_size = (3, 3),
                       padding = 'same',
                       name = 'arm_loc_%d'%i)
            )

            self.odm_cls.append(
                Conv2D(self.boxes_per_cell * num_classes,
                       kernel_size = (3, 3),
                       padding = 'same',
                       name = 'odm_cls_%d'%i)
            )
            self.odm_loc.append(
                Conv2D(self.boxes_per_cell * 4,
                       kernel_size = (3, 3),
                       padding = 'same',
                       name = 'odm_loc_%d'%i)
            )

            self.tcb_block1.append(Sequential([
                Conv2D(256,
                       kernel_size = (3, 3),
                       padding = 'same',
                       name = 'tcb_%d_block1_conv1'%i),
                #Activation('relu', name = 'tcb_%d_block1_relu'%i),
                ReLU(name = 'tcb_%d_block1_relu'%i),
                Conv2D(256,
                       kernel_size = (3, 3),
                       padding = 'same',
                       name = 'tcb_%d_block1_conv2'%i)
            ], name = 'tcb_%d_block1'%i))

            self.tcb_block2.append(Sequential([
                #Activation('relu', name = 'tcb_%d_block2_relu1'%i),
                ReLU(name = 'tcb_%d_block2_relu1'%i),
                Conv2D(256,
                       kernel_size = (3, 3),
                       padding = 'same',
                       name = 'tcb_%d_block2_conv'%i),
                #Activation('relu', name = 'tcb_%d_block2_relu2'%i)
                ReLU(name = 'tcb_%d_block2_relu2'%i)
            ], name = 'tcb_%d_block2'%i))

            if i+1 < self.num_output_features:
                self.tcb_upsample.append(
                    Conv2DTranspose(256,
                                    kernel_size = (2, 2),
                                    strides = (2, 2),
                                    name = 'tcb_%d_upsample'%i)
                )


    def call(self, x, decode = False, anchors = None):
        image_shape = x.shape[1:].as_list()

        feature4_3, feature5_3, feature7 = self.base_model(x)
        extra_feature1 = self.extra_conv1(feature7)
        extra_feature2 = self.extra_conv2(ZeroPadding2D(1)(extra_feature1))

        source_features = [
            self.conv4_3_norm(feature4_3),
            self.conv5_3_norm(feature5_3),
            feature7,
            extra_feature2
        ]

        self.feature_shapes = []
        for f in source_features:
            self.feature_shapes.append((f.shape[1].value, f.shape[2].value))

        arm_cls_features = []
        arm_loc_features = []

        for i, x in enumerate(source_features):
            arm_cls_features.append(Reshape((-1, 2))(self.arm_cls[i](x)))
            arm_loc_features.append(Reshape((-1, 4))(self.arm_loc[i](x)))

        arm_cls_features = Concatenate(axis = 1)(arm_cls_features)
        arm_loc_features = Concatenate(axis = 1)(arm_loc_features)

        tcb_features = []

        for i in range(len(source_features))[::-1]:
            x = source_features[i]

            x = self.tcb_block1[i](x)
            if len(tcb_features) > 0:
                x += self.tcb_upsample[i](tcb_features[-1])
            x = self.tcb_block2[i](x)

            tcb_features.append(x)
        tcb_features.reverse()

        odm_cls_features = []
        odm_loc_features = []

        for i, x in enumerate(tcb_features):
            odm_cls_features.append(Reshape((-1, self.num_classes))(self.odm_cls[i](x)))
            odm_loc_features.append(Reshape((-1, 4))(self.odm_loc[i](x)))

        odm_cls_features = Concatenate(axis = 1)(odm_cls_features)
        odm_loc_features = Concatenate(axis = 1)(odm_loc_features)

        if decode:
            if anchors is None:
                anchors = self.build_anchors(image_shape)

            return self.decode_output(
                arm_cls_features, arm_loc_features,
                odm_cls_features, odm_loc_features,
                anchors
            )

        return arm_cls_features, arm_loc_features, odm_cls_features, odm_loc_features

    def build_anchors(self, input_shape):
        input_height, input_width = input_shape[:2]
        anchors = []

        for scale, (feature_height, feature_width) in zip(self.scales, self.feature_shapes):
            cy, cx = np.meshgrid(range(feature_height), range(feature_width), indexing = 'ij')
            cx = (cx.astype('float32') + 0.5) * input_width / feature_width
            cy = (cy.astype('float32') + 0.5) * input_height / feature_height

            cx = np.expand_dims(cx, axis = -1)
            cy = np.expand_dims(cy, axis = -1)
            cx = np.repeat(cx, repeats = self.boxes_per_cell, axis = -1)
            cy = np.repeat(cy, repeats = self.boxes_per_cell, axis = -1)

            width = np.zeros_like(cx)
            height = np.zeros_like(cx)

            for i in range(self.boxes_per_cell):
                width[..., i] = input_width * scale * np.sqrt(self.aspect_ratios[i])
                height[..., i] = input_height * scale / np.sqrt(self.aspect_ratios[i])

            width = np.clip(width, 0, input_width)
            height = np.clip(height, 0, input_height)

            feature_anchors = np.stack([cx, cy, width, height], axis = -1)
            feature_anchors = feature_anchors.reshape(-1, 4)

            anchors.append(feature_anchors)

        return np.concatenate(anchors, axis = 0)

    def decode_output(self, arm_cls, arm_loc, odm_cls, odm_loc, anchors):
        arm_loc *= self.variances
        refined_anchor_cx = arm_loc[..., 0] * (anchors[..., 2] + K.epsilon()) + anchors[..., 0]
        refined_anchor_cy = arm_loc[..., 1] * (anchors[..., 3] + K.epsilon()) + anchors[..., 1]
        refined_anchor_w = tf.exp(arm_loc[..., 2]) * (anchors[..., 2] + K.epsilon())
        refined_anchor_h = tf.exp(arm_loc[..., 3]) * (anchors[..., 3] + K.epsilon())
        refined_anchors = tf.stack(
            [refined_anchor_cx, refined_anchor_cy, refined_anchor_w, refined_anchor_h],
            axis = -1
        )

        arm_cls = tf.nn.softmax(arm_cls, axis = -1)
        ignore = arm_cls[..., self.background_id] > self.anchor_refinement_threshold
        ignore = tf.tile(tf.expand_dims(ignore, axis=-1), [1, 1, self.num_classes])
        odm_cls *= 1 - tf.cast(ignore, tf.float32)

        detected_boxes = output_encoder.decode((odm_cls, odm_loc),
                                                 anchors = refined_anchors,
                                                 background_id = self.background_id,
                                                 conf_threshold = self.conf_threshold,
                                                 nms_threshold = self.nms_threshold,
                                                 variances = self.variances)

        return self.NMS(detected_boxes)

    def NMS(self, boxes_batch):
        output_boxes = []

        for boxes in boxes_batch:
            if len(boxes) == 0:
                output_boxes.append(tf.zeros((0, 6)))
                continue

            order = tf.argsort(boxes[:, 5], direction = 'DESCENDING')
            boxes = tf.gather_nd(boxes, indices = tf.expand_dims(order, axis = -1))

            nms_boxes = {}

            for box in boxes:
                cls = int(box[4])

                if cls not in nms_boxes:
                    nms_boxes[cls] = [box]
                    continue

                ious = IOU(box, tf.convert_to_tensor(nms_boxes[cls]))

                if np.all(ious < self.nms_threshold):
                    nms_boxes[cls].append(box)

            for cls in nms_boxes.keys():
                if len(nms_boxes[cls]) > self.max_preds_per_class:
                    nms_boxes[cls] = nms_boxes[cls][:self.max_preds_per_class]

            nms_boxes = [tf.stack(nms_boxes[cls], axis=0) for cls in nms_boxes.keys()]
            output_boxes.append(tf.concat(nms_boxes, axis = 0))

        return output_boxes
