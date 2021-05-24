import tensorflow as tf

from .utils import minmax2locenc, locenc2minmax, IOU, NMS


def encode(boxes_batch, anchors, num_classes=None,
           pos_iou_threshold=0.5,
           neg_iou_threshold=0.5,
           variances=[0.1, 0.1, 0.2, 0.2]):

    batch_size = boxes_batch.shape[0]
    num_anchors = anchors.shape[-2]
    ious = IOU(boxes_batch, anchors)

    def _cond(idx, batch_cls_array, batch_loc_array):
        return idx < batch_size

    def _loop_body(idx, batch_cls_array, batch_loc_array):
        # Initialize the encoded tensors
        cls = tf.zeros((num_anchors, num_classes), dtype=tf.float32)
        loc = tf.zeros((num_anchors, 4), dtype=tf.float32)

        boxes = boxes_batch[idx]
        batch_ious = ious[idx]
        num_boxes = tf.shape(boxes)[0]

        best_match_per_gt = tf.argmax(batch_ious, axis=1, output_type=tf.int32)

        # Ensure the best anchor for each ground truth will be picked
        indices = tf.stack([tf.range(num_boxes), best_match_per_gt], axis=1)
        updates = tf.zeros(tf.shape(boxes)[:1], dtype=tf.float32) + 2.0
        batch_ious = tf.tensor_scatter_nd_update(batch_ious, indices, updates)

        # Then find the best ground truth for each anchor
        max_iou_per_anchor = tf.reduce_max(batch_ious, axis=0)
        best_match_per_anchor = tf.argmax(batch_ious, axis=0,
                                          output_type=tf.int32)

        # Identify which anchors are positives/negatives
        positive = max_iou_per_anchor >= pos_iou_threshold
        negative = max_iou_per_anchor < neg_iou_threshold
        pos_idxs = tf.boolean_mask(tf.range(num_anchors), mask=positive)
        neg_idxs = tf.boolean_mask(tf.range(num_anchors), mask=negative)

        # Get the positive anchors in the right format
        if len(anchors.shape) == 2:
            pos_anchors = tf.gather(anchors, indices=pos_idxs, axis=0)
        else:
            pos_anchors = tf.gather(anchors[idx], indices=pos_idxs, axis=0)

        # Same for the matched boxes for each of these positive anchors
        matches = tf.gather(best_match_per_anchor, indices=pos_idxs, axis=0)
        pos_boxes = tf.gather(boxes, indices=matches, axis=0)
        labels = pos_boxes[..., 4]

        # Setting the negatives
        indices = tf.stack([neg_idxs, tf.zeros_like(neg_idxs)], axis=1)
        updates = tf.ones_like(neg_idxs, dtype=tf.float32)
        cls = tf.tensor_scatter_nd_update(cls, indices, updates)

        # Setting the positives
        indices = tf.stack([pos_idxs, tf.cast(labels, tf.int32)], axis=1)
        updates = tf.ones_like(pos_idxs, dtype=tf.float32)
        cls = tf.tensor_scatter_nd_update(cls, indices, updates)

        # Setting the localization encoding
        indices = tf.expand_dims(pos_idxs, -1)
        updates = minmax2locenc(pos_boxes, pos_anchors, variances)
        loc = tf.tensor_scatter_nd_update(loc, indices, updates)

        # Add to the encoded batch
        batch_cls_array = batch_cls_array.write(idx, cls)
        batch_loc_array = batch_loc_array.write(idx, loc)

        return idx + 1, batch_cls_array, batch_loc_array

    batch_cls_array = tf.TensorArray(
        tf.float32, size=batch_size, dynamic_size=False, clear_after_read=True)
    batch_loc_array = tf.TensorArray(
        tf.float32, size=batch_size, dynamic_size=False, clear_after_read=True)

    _, batch_cls_array, batch_loc_array = tf.while_loop(
        _cond, _loop_body, [tf.constant(0), batch_cls_array, batch_loc_array])

    batch_cls = batch_cls_array.stack()
    batch_loc = batch_loc_array.stack()

    return batch_cls, batch_loc


def decode(encoded_output, anchors,
           conf_threshold=0.5, nms_threshold=0.45,
           max_preds_per_class=100,
           variances=[0.1, 0.1, 0.2, 0.2]):

    batch_cls, batch_loc = encoded_output
    batch_size = batch_cls.shape[0]

    boxes_batch = []

    for idx in range(batch_size):
        cls = batch_cls[idx]
        loc = batch_loc[idx]

        num_classes = cls.shape[-1]
        boxes = []

        for label in range(1, num_classes):
            conf = cls[..., label]
            positives = conf >= conf_threshold

            class_conf = tf.boolean_mask(conf, mask=positives, axis=0)
            class_loc = tf.boolean_mask(loc, mask=positives, axis=0)

            if len(anchors.shape) == 2:
                pos_anchors = tf.boolean_mask(
                    anchors, mask=positives, axis=0)
            else:
                pos_anchors = tf.boolean_mask(
                    anchors[idx], mask=positives, axis=0)

            class_boxes = locenc2minmax(class_loc, pos_anchors, variances)
            class_conf = tf.expand_dims(class_conf, axis=-1)

            class_boxes = tf.concat([class_boxes,
                                     tf.zeros_like(class_conf) + label,
                                     class_conf], axis=-1)

            boxes.append(NMS(class_boxes, nms_threshold=nms_threshold,
                             top_k=max_preds_per_class))

        boxes_batch.append(tf.concat(boxes, axis=0))

    boxes_batch = tf.RaggedTensor.from_row_lengths(
        values=tf.concat(boxes_batch, axis=0),
        row_lengths=[tf.shape(boxes)[0] for boxes in boxes_batch]
    )

    return boxes_batch
