IMAGE_SIZE = (320, 320)
BATCH_SIZE = 32
SHUFFLE_BUFFER = 500
NUM_EPOCHS = 24
STEPS_PER_EPOCH = 5000
LR_SCHEDULE = ([80000, 100000], [1e-3, 1e-4, 1e-5])
MOMENTUM = 0.9

VOC_CLASSES = [
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
