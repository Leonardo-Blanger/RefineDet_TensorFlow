import tensorflow as tf

from losses import RefineDetLoss
import output_encoder
from utils import locenc2minmax


class RefineDetBase(tf.keras.Model):
    def __init__(self,
                 conf_threshold=0.01,
                 anchor_refinement_threshold=0.99,
                 max_preds_per_class=200,
                 nms_threshold=0.45,
                 neg_to_pos_ratio=3,
                 pos_iou_threshold=0.5,
                 neg_iou_threshold=0.5,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 **kwargs):

        self.conf_threshold = conf_threshold
        self.anchor_refinement_threshold = anchor_refinement_threshold
        self.nms_threshold = nms_threshold
        self.max_preds_per_class = max_preds_per_class
        self.variances = variances

        self.refinedet_loss = RefineDetLoss(
            num_classes = self.num_classes,
            anchor_refinement_threshold = anchor_refinement_threshold,
            neg_to_pos_ratio=neg_to_pos_ratio,
            pos_iou_threshold = pos_iou_threshold,
            neg_iou_threshold = neg_iou_threshold,
            variances = variances)

        super(RefineDetBase, self).__init__(**kwargs)


    def train_step(self, data):
        x, y_true = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            
            all_losses = self.refinedet_loss(y_true, y_pred)
            arm_cls_loss, arm_loc_loss, odm_cls_loss, odm_loc_loss = [
                all_losses[i] for i in range(4)
            ]
            l2_reg_loss = tf.reduce_sum(self.losses)
            loss = arm_cls_loss + arm_loc_loss + odm_cls_loss + odm_loc_loss + l2_reg_loss

        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return {'arm_cls_loss': arm_cls_loss,
                'arm_loc_loss': arm_loc_loss,
                'odm_cls_loss': odm_cls_loss,
                'odm_loc_loss': odm_loc_loss,
                'l2_reg_loss': l2_reg_loss,
                'total_loss': loss}


    def decode(self, output):
        (arm_cls, arm_loc), (odm_cls, odm_loc), anchors = output
        refined_anchors = locenc2minmax(arm_loc, anchors, self.variances)

        ignore = arm_cls[..., :1] >= self.anchor_refinement_threshold # not easy negatives
        odm_cls *= 1.0 - tf.cast(ignore, tf.float32)

        return output_encoder.decode((odm_cls, odm_loc), refined_anchors,
                                     self.conf_threshold, self.nms_threshold,
                                     self.max_preds_per_class, self.variances)
