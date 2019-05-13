image_size = 320
channel_means = [104, 117, 123]
pos_iou_threshold = 0.5
neg_iou_threshold = 0.5
anchor_refinement_threshold = 0.99
variances = [0.1, 0.1, 0.2, 0.2]

batch_size = 32
num_iterations = 120000
learning_rates = [(0, 1e-3), (80000, 1e-4), (100000, 1e-5)]
momentum = 0.9
weight_decay = 5e-4

voc_classes = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

classes = voc_classes
