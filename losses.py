import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy

import output_encoder
from utils import minmax2xywh, xywh2minmax, locenc2minmax


def smooth_l1(a, b):
    x = tf.abs(a - b)
    return tf.where(x < 1, 0.5*x*x, x - 0.5)


class RefineDetLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes,
                 anchor_refinement_threshold=0.99,
                 neg_to_pos_ratio=3,
                 pos_iou_threshold=0.5,
                 neg_iou_threshold=0.5,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 **kwargs):

        self.num_classes = num_classes
        self.anchor_refinement_threshold = anchor_refinement_threshold
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.variances = variances

        self.encoding_params = {
            'pos_iou_threshold': pos_iou_threshold,
            'neg_iou_threshold': neg_iou_threshold,
            'variances': variances}

        super().__init__(reduction=tf.keras.losses.Reduction.NONE, **kwargs)


    def get_config(self):
        base_config = super().get_config()
        return {**base_config, **self.encoding_params,
                'num_classes': self.num_classes,
                'anchor_refinement_threshold': self.anchor_refinement_threshold,
                'neg_to_pos_ratio': self.neg_to_pos_ratio}


    @tf.function
    def call(self, y_true, y_pred):
        (arm_cls, arm_loc), (odm_cls, odm_loc), anchors = y_pred

        binary_y_true = tf.concat([y_true[...,:4],
                                   tf.ones_like(y_true[...,4:5])], axis=-1)

        true_cls, true_loc = output_encoder.encode(binary_y_true, anchors,
                                                   num_classes=2, **self.encoding_params)
        true_cls = tf.stop_gradient(true_cls)
        true_loc = tf.stop_gradient(true_loc)

        arm_cls_loss, arm_loc_loss = self.ssd_loss((true_cls, true_loc),
                                                   (arm_cls, arm_loc))

        refined_anchors = locenc2minmax(arm_loc, anchors, self.variances)

        refined_cls, refined_loc = output_encoder.encode(y_true, refined_anchors,
                                                         num_classes=self.num_classes,
                                                         **self.encoding_params)

        ignore = arm_cls[..., :1] < self.anchor_refinement_threshold # not easy negatives
        ignore = tf.cast(ignore, tf.float32)
        refined_cls *= ignore

        refined_cls = tf.stop_gradient(refined_cls)
        refined_loc = tf.stop_gradient(refined_loc)

        odm_cls_loss, odm_loc_loss = self.ssd_loss((refined_cls, refined_loc),
                                                   (odm_cls, odm_loc))

        return arm_cls_loss, arm_loc_loss, odm_cls_loss, odm_loc_loss


    def ssd_loss(self, y_true, y_pred):
        cls_true, loc_true = y_true
        cls_pred, loc_pred = y_pred

        batch_size = tf.shape(cls_true)[0]
        num_anchors = tf.shape(cls_true)[1]
        
        negative_mask = cls_true[..., 0]
        positive_mask = tf.reduce_sum(cls_true[..., 1:], axis=-1)
        N = tf.reduce_sum(positive_mask)

        cls_loss = categorical_crossentropy(
            y_true=cls_true, y_pred=cls_pred, from_logits=False)
        loc_loss = positive_mask * tf.reduce_sum(
            smooth_l1(loc_true, loc_pred), axis=-1)

        pos_cls_loss = positive_mask * cls_loss
        neg_cls_loss = negative_mask * cls_loss
        
        num_negatives_to_keep = tf.minimum(
            tf.reduce_sum(negative_mask),
            N * self.neg_to_pos_ratio
        )
        num_negatives_to_keep = tf.cast(num_negatives_to_keep, tf.int32)

        neg_cls_loss = tf.reshape(neg_cls_loss, [-1])
        neg_cls_loss = tf.sort(neg_cls_loss)[-num_negatives_to_keep:]

        cls_loss = (tf.reduce_sum(pos_cls_loss) + tf.reduce_sum(neg_cls_loss)) / N
        loc_loss = tf.reduce_sum(loc_loss) / N

        return cls_loss, loc_loss


