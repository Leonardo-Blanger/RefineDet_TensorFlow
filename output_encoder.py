import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import tensorflow.keras.backend as K

cx, cy, w, h = 0, 1, 2, 3

def anchor_IOU(boxes, anchors):
    boxes = tf.tile(tf.expand_dims(boxes, 1), [1, anchors.shape[0], 1])

    anchor_xmin = anchors[:, cx] - anchors[:, w] * 0.5
    anchor_ymin = anchors[:, cy] - anchors[:, h] * 0.5
    anchor_xmax = anchors[:, cx] + anchors[:, w] * 0.5
    anchor_ymax = anchors[:, cy] + anchors[:, h] * 0.5

    anchor_ymin = tf.tile(tf.expand_dims(anchor_ymin, 0), [boxes.shape[0], 1])
    anchor_xmin = tf.tile(tf.expand_dims(anchor_xmin, 0), [boxes.shape[0], 1])
    anchor_xmax = tf.tile(tf.expand_dims(anchor_xmax, 0), [boxes.shape[0], 1])
    anchor_ymax = tf.tile(tf.expand_dims(anchor_ymax, 0), [boxes.shape[0], 1])

    inter_w = tf.maximum(0, tf.minimum(boxes[..., 2], anchor_xmax) - tf.maximum(boxes[..., 0], anchor_xmin))
    inter_h = tf.maximum(0, tf.minimum(boxes[..., 3], anchor_ymax) - tf.maximum(boxes[..., 1], anchor_ymin))
    intersection = inter_w * inter_h

    area_boxes = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    area_anchors = (anchor_xmax - anchor_xmin) * (anchor_ymax - anchor_ymin)
    union = area_boxes + area_anchors - intersection

    return intersection / union


def encode(boxes_batch, anchors, num_classes, background_id = 0,
                            pos_iou_threshold = 0.5, neg_iou_threshold = 0.5,
                            variances = [0.1, 0.1, 0.2, 0.2]):
    batch_size = len(boxes_batch)
    num_anchors = anchors.shape[-2]

    encoded_cls = np.zeros((batch_size, num_anchors, num_classes), dtype='float32')
    encoded_loc = np.zeros((batch_size, num_anchors, 4), dtype='float32')

    for batch_idx, boxes in enumerate(boxes_batch):
        if anchors.ndim == 2:
            ious = anchor_IOU(boxes, anchors).numpy()
        else:
            ious = anchor_IOU(boxes, anchors[batch_idx]).numpy()

        best_anchor_for_gt = np.zeros((len(boxes),), dtype='int32')

        # First find the best anchor for each gt
        def find_best_anchor_for_gt(ious):
            for _ in range(len(boxes)):
                best_gt = tf.argmax(tf.reduce_max(ious, axis = 1))
                best_anchor = tf.argmax(ious[best_gt])

                best_anchor_for_gt[best_gt] = best_anchor
                ious[best_gt, :] = -1.0
                ious[:, best_anchor] = -1.0

        find_best_anchor_for_gt(ious.copy())
        ious[tf.range(len(boxes)), best_anchor_for_gt] = 2

        best_iou = tf.reduce_max(ious, axis = 0)
        best_gt_for_anchor = tf.argmax(ious, axis = 0)

        pos_mask = (best_iou >= pos_iou_threshold).numpy()
        neg_mask = (best_iou < neg_iou_threshold).numpy()

        encoded_cls[batch_idx, neg_mask, background_id] = 1.0

        boxes_cx = (boxes[:,0] + boxes[:,2]) * 0.5
        boxes_cy = (boxes[:,1] + boxes[:,3]) * 0.5
        boxes_w = boxes[:,2] - boxes[:,0]
        boxes_h = boxes[:,3] - boxes[:,1]
        boxes_label = boxes[:,4]

        matched_gt_idx = tf.boolean_mask(best_gt_for_anchor, mask=pos_mask)

        boxes_cx = tf.gather(boxes_cx, indices = matched_gt_idx)
        boxes_cy = tf.gather(boxes_cy, indices = matched_gt_idx)
        boxes_w = tf.gather(boxes_w, indices = matched_gt_idx)
        boxes_h = tf.gather(boxes_h, indices = matched_gt_idx)
        boxes_label = tf.gather(boxes_label, indices = matched_gt_idx)

        if anchors.ndim == 2:
            pos_anchors = tf.boolean_mask(anchors, mask=pos_mask)
        else:
            pos_anchors = tf.boolean_mask(anchors[batch_idx], mask=pos_mask)

        anchor_cx = pos_anchors[:, cx]
        anchor_cy = pos_anchors[:, cy]
        anchor_w = pos_anchors[:, w]
        anchor_h = pos_anchors[:, h]

        encoded_cls[batch_idx, pos_mask, tf.cast(boxes_label, tf.int32)] = 1.0

        encoded_loc[batch_idx, pos_mask, 0] = (boxes_cx - anchor_cx) / (anchor_w + K.epsilon())
        encoded_loc[batch_idx, pos_mask, 1] = (boxes_cy - anchor_cy) / (anchor_h + K.epsilon())
        encoded_loc[batch_idx, pos_mask, 2] = tf.log(boxes_w / (anchor_w + K.epsilon()))
        encoded_loc[batch_idx, pos_mask, 3] = tf.log(boxes_h / (anchor_h + K.epsilon()))
        encoded_loc[batch_idx, pos_mask, :] /= variances

    return encoded_cls, encoded_loc


def decode(output, anchors, background_id = 0, conf_threshold = 0.5,
                    nms_threshold = 0.5, variances = [0.1, 0.1, 0.2, 0.2]):

    cls, loc = output
    cls = tf.nn.softmax(cls, axis = -1)
    batch_size = cls.shape[0]
    variances = np.array(variances)

    positives = tf.logical_and(
        cls[..., background_id] < conf_threshold,
        tf.reduce_max(cls, axis = -1) >= conf_threshold
    )

    boxes_batch = []

    for batch_idx in range(batch_size):
        pos_cls = tf.boolean_mask(cls[batch_idx], mask = positives[batch_idx])
        pos_loc = tf.boolean_mask(loc[batch_idx], mask = positives[batch_idx])

        if anchors.ndim == loc.ndim:
            pos_anchors = tf.boolean_mask(anchors[batch_idx], mask = positives[batch_idx])
        else:
            pos_anchors = tf.boolean_mask(anchors, mask = positives[batch_idx])

        label = tf.cast(tf.argmax(pos_cls, axis = -1), tf.float32)
        conf = tf.reduce_max(pos_cls, axis = -1)

        pos_loc *= variances
        cx = pos_loc[..., 0] * (pos_anchors[..., 2] + K.epsilon()) + pos_anchors[..., 0]
        cy = pos_loc[..., 1] * (pos_anchors[..., 3] + K.epsilon()) + pos_anchors[..., 1]
        width = tf.exp(pos_loc[..., 2]) * (pos_anchors[..., 2] + K.epsilon())
        height = tf.exp(pos_loc[..., 3]) * (pos_anchors[..., 3] + K.epsilon())

        xmin = cx - width * 0.5
        ymin = cy - height * 0.5
        xmax = cx + width * 0.5
        ymax = cy + height * 0.5

        boxes = tf.stack([xmin, ymin, xmax, ymax, label, conf], axis = -1)

        boxes_batch.append(boxes)

    return boxes_batch
