import tensorflow as tf
from tensorflow.keras import backend as K


@tf.function(experimental_relax_shapes=True)
def read_jpeg_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


@tf.function(experimental_relax_shapes=True)
def resize_image_and_boxes(image, boxes, new_size=(320, 320)):
    prev_height = tf.shape(image)[0]
    prev_width = tf.shape(image)[1]
    image = tf.image.resize(image, new_size[:2])
    boxes = tf.cast(boxes, tf.int32)

    xmins = boxes[..., 0] * new_size[1] / prev_width
    ymins = boxes[..., 1] * new_size[0] / prev_height
    xmaxs = boxes[..., 2] * new_size[1] / prev_width
    ymaxs = boxes[..., 3] * new_size[0] / prev_height

    new_boxes = tf.stack([xmins, ymins, xmaxs, ymaxs], axis=1)
    boxes = tf.concat([tf.cast(new_boxes, tf.float32),
                       tf.cast(boxes[..., 4:], tf.float32)], axis=1)

    return image, boxes


@tf.function(experimental_relax_shapes=True)
def absolute2relative(boxes, size):
    boxes = tf.cast(boxes, tf.float32)
    xmins = boxes[..., 0] / tf.cast(size[1], tf.float32)
    ymins = boxes[..., 1] / tf.cast(size[0], tf.float32)
    xmaxs = boxes[..., 2] / tf.cast(size[1], tf.float32)
    ymaxs = boxes[..., 3] / tf.cast(size[0], tf.float32)

    new_boxes = tf.stack([xmins, ymins, xmaxs, ymaxs], axis=1)
    boxes = tf.concat([new_boxes, boxes[..., 4:]], axis=1)

    return boxes


@tf.function(experimental_relax_shapes=True)
def minmax2xywh(boxes):
    xmin, ymin, xmax, ymax = [boxes[..., i] for i in range(4)]
    cx = (xmin + xmax)*0.5
    cy = (ymin + ymax)*0.5
    w = xmax - xmin
    h = ymax - ymin
    new_boxes = tf.stack([cx, cy, w, h], axis=-1)
    return tf.concat([new_boxes, boxes[..., 4:]], axis=-1)


@tf.function(experimental_relax_shapes=True)
def xywh2minmax(boxes):
    cx, cy, w, h = [boxes[..., i] for i in range(4)]
    xmin = cx - w*0.5
    ymin = cy - h*0.5
    xmax = cx + w*0.5
    ymax = cy + h*0.5
    new_boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    return tf.concat([new_boxes, boxes[..., 4:]], axis=-1)


@tf.function(experimental_relax_shapes=True)
def locenc2xywh(loc, anchors, variances):
    anchors_xywh = minmax2xywh(anchors)
    anchor_cx, anchor_cy, anchor_w, anchor_h = [anchors_xywh[..., i]
                                                for i in range(4)]

    loc = loc * variances
    cx = loc[..., 0] * (anchor_w + K.epsilon()) + anchor_cx
    cy = loc[..., 1] * (anchor_h + K.epsilon()) + anchor_cy
    w = tf.math.exp(loc[..., 2]) * (anchor_w + K.epsilon())
    h = tf.math.exp(loc[..., 3]) * (anchor_h + K.epsilon())
    boxes_xywh = tf.stack([cx, cy, w, h], axis=-1)

    return boxes_xywh


@tf.function(experimental_relax_shapes=True)
def locenc2minmax(loc, anchors, variances):
    return xywh2minmax(locenc2xywh(loc, anchors, variances))


@tf.function(experimental_relax_shapes=True)
def xywh2locenc(boxes, anchors, variances):
    anchors_xywh = minmax2xywh(anchors)
    anchor_cx, anchor_cy, anchor_w, anchor_h = [anchors_xywh[..., i]
                                                for i in range(4)]
    box_cx, box_cy, box_w, box_h, labels = [boxes[...,  i]
                                            for i in range(5)]

    loc_cx = (box_cx - anchor_cx) / (anchor_w + K.epsilon())
    loc_cy = (box_cy - anchor_cy) / (anchor_h + K.epsilon())
    loc_w = tf.math.log(box_w / (anchor_w + K.epsilon()))
    loc_h = tf.math.log(box_h / (anchor_h + K.epsilon()))
    loc = tf.stack([loc_cx, loc_cy, loc_w, loc_h], axis=-1)
    loc = loc / variances

    return loc


@tf.function(experimental_relax_shapes=True)
def minmax2locenc(boxes, anchors, variances):
    return xywh2locenc(minmax2xywh(boxes), anchors, variances)


@tf.function(experimental_relax_shapes=True)
def IOU(boxes1, boxes2):
    # boxes1.shape: (B, (N), 4)
    # boxes2.shape: (M, 4) or (B, M, 4)

    boxes1 = tf.expand_dims(boxes1, axis=-2)
    boxes2 = tf.expand_dims(boxes2, axis=-3)
    # boxes1.shape: (B, (N), 1, 4)
    # boxes2.shape: (1, M, 4) or (B, 1, M, 4)

    boxes1_width = (boxes1[..., 2] - boxes1[..., 0])
    boxes1_height = (boxes1[..., 3] - boxes1[..., 1])
    boxes1_area = boxes1_width * boxes1_height
    # boxes1_area.shape: (B, (N), 1)

    boxes2_width = (boxes2[..., 2] - boxes2[..., 0])
    boxes2_height = (boxes2[..., 3] - boxes2[..., 1])
    boxes2_area = boxes2_width * boxes2_height
    # boxes2_area.shape: (1, M) or (B, 1, M)

    inter_xmin = tf.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_xmax = tf.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_width = tf.maximum(inter_xmax - inter_xmin, 0.0)

    inter_ymin = tf.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_ymax = tf.minimum(boxes1[..., 3], boxes2[..., 3])
    inter_height = tf.maximum(inter_ymax - inter_ymin, 0.0)

    inter_area = inter_width * inter_height
    union_area = boxes1_area + boxes2_area - inter_area

    # return.shape: (B, (N), M)
    return inter_area / union_area


@tf.function(experimental_relax_shapes=True)
def NMS(boxes, top_k=100, nms_threshold=0.45):
    num_boxes = tf.minimum(top_k, tf.shape(boxes)[0])
    _, idxs = tf.nn.top_k(boxes[:, 5], k=num_boxes)
    boxes = tf.gather_nd(boxes, indices=tf.expand_dims(idxs, axis=-1))

    ious = IOU(boxes, boxes)
    
    def cond(idx, boxes):
        return idx < num_boxes

    def loop(idx, boxes):
        if boxes[idx,4] == 0.0:
            return idx+1, boxes

        suppress = tf.logical_and(
            tf.range(num_boxes) > idx,
            ious[idx] >= nms_threshold)
        
        indices = tf.boolean_mask(tf.range(num_boxes), mask=suppress, axis=0)
        indices = tf.expand_dims(indices, -1)
        updates = tf.zeros((tf.shape(indices)[0], tf.shape(boxes)[1]))
        boxes = tf.tensor_scatter_nd_update(boxes, indices, updates)

        return idx+1, boxes

    _, boxes = tf.while_loop(cond, loop, [tf.constant(0), boxes])

    remain = boxes[:, 4] > 0.0
    boxes = tf.boolean_mask(boxes, mask=remain, axis=0)

    return boxes


@tf.function(experimental_relax_shapes=True)
def clip_boxes(boxes, x0, y0, x1, y1):
    xmin, ymin, xmax, ymax = [boxes[:, i] for i in range(4)]

    x0 = tf.cast(x0, boxes.dtype)
    y0 = tf.cast(y0, boxes.dtype)
    x1 = tf.cast(x1, boxes.dtype)
    y1 = tf.cast(y1, boxes.dtype)

    xmin = tf.clip_by_value(xmin, x0, x1)
    ymin = tf.clip_by_value(ymin, y0, y1)
    xmax = tf.clip_by_value(xmax, x0, x1)
    ymax = tf.clip_by_value(ymax, y0, y1)
    stacked_boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    boxes = tf.concat([stacked_boxes, boxes[:, 4:]], axis=-1)

    return boxes

