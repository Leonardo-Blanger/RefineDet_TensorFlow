import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .utils import IOU


class MeanAveragePrecision(tf.keras.metrics.Metric):
    def __init__(self, iou_threshold=0.5, **kwargs):
        super().__init__(dynamic=True, **kwargs)
        self.iou_threshold = iou_threshold
        self.ground_truth = []
        self.predictions = []
        self.ignore = []

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'iou_threshold': self.iou_threshold}

    def update_state(self, y_true, y_pred, ignore_samples=None):
        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            yt, yp = np.array(yt), np.array(yp)
            self.ground_truth.append(yt)
            self.predictions.append(yp)

            if ignore_samples is None:
                self.ignore.append(np.zeros((len(yt),)))
            else:
                self.ignore.append(np.array(ignore_samples[i]))

    def reset_state(self):
        self.ground_truth = []
        self.predictions = []
        self.ignore = []

    def result(self):
        return np.mean(self.per_class_AP())

    def per_class_AP(self):
        classes = np.unique(np.concatenate([np.unique(boxes[:, 4])
                                            for boxes in self.ground_truth]))
        APs = []

        for cls in classes:
            print('\nCalculating Average Precision for class', cls)

            class_ground_truth = [
                boxes[boxes[:, 4] == cls] for boxes in self.ground_truth]
            class_predictions = [
                boxes[boxes[:, 4] == cls] for boxes in self.predictions]
            class_ignore = [
                ignore[boxes[:, 4] == cls] for ignore, boxes in zip(self.ignore, self.ground_truth)
            ]

            APs.append(self.AP(class_ground_truth, class_predictions, class_ignore))

        return APs

    def AP(self, batch_ground_truth, batch_predictions, batch_ignore):
        all_predictions = []
        total_positives = 0

        for ground_truth, predictions, ignore in tqdm(
                zip(batch_ground_truth, batch_predictions, batch_ignore),
                desc='Calculating Average Precision...'):

            total_positives += len(ground_truth) - np.sum(ignore)

            matched = np.zeros(len(ground_truth))
            predictions = sorted(predictions, key=lambda x: x[5], reverse=True)

            for pred_idx, pred in enumerate(predictions):
                if len(ground_truth) == 0:
                    all_predictions.append((pred[5], False))
                    continue

                iou = IOU([pred], ground_truth)[0]
                i = np.argmax(iou)

                if not ignore[i]:
                    if iou[i] >= self.iou_threshold and not matched[i]:
                        all_predictions.append((pred[5], True))
                        matched[i] = True
                    else:
                        all_predictions.append((pred[5], False))

        all_predictions = sorted(all_predictions, reverse=True)

        recalls, precisions = [0], [1]
        TPs, FPs = 0, 0

        for _, result in all_predictions:
            if result:
                TPs += 1
            else:
                FPs += 1
            precisions.append(TPs / (TPs+FPs))
            recalls.append(TPs / total_positives)

        recalls = np.array(recalls)
        precisions = np.array(precisions)

        for i in range(len(precisions)-2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i+1])

        return np.sum((recalls[1:]-recalls[:-1]) * precisions[1:])
