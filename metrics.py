import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import imgaug as ia
import tensorflow.keras.backend as K

from output_encoder import encode

def IOU(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = [box1[..., i] for i in range(4)]
    xmin2, ymin2, xmax2, ymax2 = [box2[..., i] for i in range(4)]

    inter_width  = tf.maximum(0, tf.minimum(xmax1, xmax2) - tf.maximum(xmin1, xmin2))
    inter_height = tf.maximum(0, tf.minimum(ymax1, ymax2) - tf.maximum(ymin1, ymin2))
    intersection = inter_width * inter_height
    union = (xmax1-xmin1)*(ymax1-ymin1) + (xmax2-xmin2)*(ymax2-ymin2) - intersection

    return intersection / union

def smooth_l1_loss(y_true, y_pred):
    x = tf.abs(y_true - y_pred)
    return tf.where(x < 1.0, 0.5*x*x, x - 0.5)


def ssd_losses(true_cls, true_loc, out_cls, out_loc, neg_to_pos_ratio = 3.0, weights = 1.0):
    neg_mask = true_cls[..., 0] * weights
    pos_mask = tf.reduce_sum(true_cls[..., 1:], axis=-1) * weights
    N = tf.reduce_sum(pos_mask)

    if N.numpy() == 0:
        return tf.convert_to_tensor(0, dtype=tf.float32), tf.convert_to_tensor(0, dtype=tf.float32)

    pos_mask = tf.cast(pos_mask, tf.bool)
    neg_mask = tf.cast(neg_mask, tf.bool)

    out_cls_pos = tf.boolean_mask(out_cls, mask=pos_mask)
    out_cls_neg = tf.boolean_mask(out_cls, mask=neg_mask)

    true_cls_pos = tf.boolean_mask(true_cls, mask=pos_mask)
    true_cls_neg = tf.boolean_mask(true_cls, mask=neg_mask)

    out_loc = tf.boolean_mask(out_loc, mask=pos_mask)
    true_loc = tf.boolean_mask(true_loc, mask=pos_mask)

    pos_cls_loss = tf.losses.softmax_cross_entropy(
        onehot_labels = true_cls_pos,
        logits = out_cls_pos,
        reduction = tf.losses.Reduction.SUM
    )

    neg_cls_loss = tf.losses.softmax_cross_entropy(
        onehot_labels = true_cls_neg,
        logits = out_cls_neg,
        reduction = tf.losses.Reduction.NONE
    )

    loc_loss = tf.reduce_sum(smooth_l1_loss(true_loc, out_loc))

    negatives_to_consider = tf.minimum(
        tf.cast(neg_to_pos_ratio * N, dtype=tf.int32),
        tf.cast(true_cls_neg.shape[0], dtype=tf.int32)
    )
    neg_cls_loss = tf.reduce_sum(tf.sort(neg_cls_loss)[-negatives_to_consider:])

    return (pos_cls_loss + neg_cls_loss) / N, loc_loss / N


def refinedet_losses(ground_truth, output, anchors, num_classes, background_id = 0,
        anchor_refinement_threshold = 0.99, variances = [0.1, 0.1, 0.2, 0.2],
        pos_iou_threshold = 0.5, neg_iou_threshold = 0.5):

    binary_ground_truth = [tf.concat(
        [gt[..., :4], tf.ones_like(gt[..., 4:])],
    axis = -1) for gt in ground_truth]

    true_cls, true_loc = encode(boxes_batch = binary_ground_truth,
                                anchors = anchors,
                                num_classes = 2,
                                background_id = background_id,
                                pos_iou_threshold = pos_iou_threshold,
                                neg_iou_threshold = neg_iou_threshold,
                                variances = variances)

    arm_cls, arm_loc, odm_cls, odm_loc = output
    variances = np.array(variances)

    arm_cls_loss, arm_loc_loss = ssd_losses(true_cls, true_loc, arm_cls, arm_loc)

    # Decode the ARM loc output to get the refined anchors for the ODM
    arm_loc *= variances
    refined_anchor_cx = arm_loc[..., 0] * (anchors[..., 2] + K.epsilon()) + anchors[..., 0]
    refined_anchor_cy = arm_loc[..., 1] * (anchors[..., 3] + K.epsilon()) + anchors[..., 1]
    refined_anchor_width = tf.exp(arm_loc[..., 2]) * (anchors[..., 2] + K.epsilon())
    refined_anchor_height = tf.exp(arm_loc[..., 3]) * (anchors[..., 3] + K.epsilon())
    refined_anchors = tf.stack(
        [refined_anchor_cx, refined_anchor_cy, refined_anchor_width, refined_anchor_height],
        axis = -1
    )

    refined_cls, refined_loc = encode(boxes_batch = ground_truth,
                                    anchors = refined_anchors,
                                    num_classes = num_classes,
                                    background_id = background_id,
                                    pos_iou_threshold = pos_iou_threshold,
                                    neg_iou_threshold = neg_iou_threshold,
                                    variances = variances)

    # Compose a weight vector to ignore the well classified negative boxes
    weights = tf.cast(
        tf.nn.softmax(arm_cls, axis=-1)[..., 0] < anchor_refinement_threshold,
        dtype = np.float32
    )

    odm_cls_loss, odm_loc_loss = ssd_losses(refined_cls, refined_loc, odm_cls, odm_loc, weights = weights)

    return arm_cls_loss, arm_loc_loss, odm_cls_loss, odm_loc_loss


def AP(batch_ground_truth, batch_predictions, iou_threshold = 0.5, version = '12'):
    all_predictions = []
    total_positives = 0

    for ground_truth, predictions in zip(batch_ground_truth, batch_predictions):
        if len(ground_truth) > 0:
            total_positives += int((1 - ground_truth[:,5]).numpy().sum())

        predictions = sorted(predictions, key=lambda x: x[5].numpy(), reverse=True)

        matched = np.zeros(len(ground_truth))

        for pred in predictions:
            if len(ground_truth) == 0:
                all_predictions.append((pred[5].numpy(), False))
                continue

            iou = IOU(pred, ground_truth)# [pred.iou(gt) for gt in ground_truth]
            i = tf.argmax(iou)

            # TODO: make this better
            if ground_truth[i,5]:
                continue

            if iou[i].numpy() >= iou_threshold:
                if not matched[i]:
                    all_predictions.append((pred[5].numpy(), True))
                    matched[i] = True
                else:
                    all_predictions.append((pred[5].numpy(), False))
            else:
                all_predictions.append((pred[5].numpy(), False))

    all_predictions = sorted(all_predictions, reverse=True)

    recalls, precisions = [0], [1]
    TP, FP = 0, 0

    for conf, result in all_predictions:
        if result: TP += 1
        else: FP += 1

        precisions.append(TP / (TP+FP))
        recalls.append(TP / total_positives)

    recalls = np.array(recalls)
    precisions = np.array(precisions)

    if version == '07':
        AP = 0
        for t in np.arange(0., 1.1, 0.1):
            if np.any(recalls >= t):
                AP += precisions[recalls >= t].max()
        AP /= 11
        return AP

    if version == '12':
        for i in range(len(precisions)-2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i+1])

        return np.sum((recalls[1:]-recalls[:-1]) * precisions[1:])

    raise Exception('Unknown version:', version)

def per_class_AP(ground_truth, predictions, iou_threshold = 0.5, version = '12'):
    classes = np.unique(np.concatenate([
        np.unique([int(box[4]) for box in boxes])
        for boxes in ground_truth
    ]))

    APs = []

    for cls in classes:
        class_ground_truth = [
            tf.convert_to_tensor([box for box in boxes if int(box[4]) == cls])
            for boxes in ground_truth
        ]
        class_predictions = [
            tf.convert_to_tensor([box for box in boxes if int(box[4]) == cls])
            for boxes in predictions
        ]

        APs.append(AP(class_ground_truth, class_predictions, iou_threshold, version))

    return APs


def meanAP(y_true, y_pred, classes, iou_threshold = 0.5):
    return np.mean(per_class_AP(y_true, y_pred, classes = classes, iou_threshold = iou_threshold))
