# RefineDet implementation with TensorFlow

This project is intended to be a TensorFlow 2.0 implementation of the RefineDet object detection model, from the paper *Single-Shot Refinement Neural Network for Object Detection* [(Zhang et al.)](https://arxiv.org/abs/1711.06897), originally made in Caffe.

I am releasing an inference demo script, as well as a weights file, trained on the 20 class PASCAL VOC 07+12 trainval datasets [(Everingham et al.)](http://host.robots.ox.ac.uk/pascal/VOC/). These weights are already achieving around 80% mean AP on the VOC 2007 test set, which is about the same as reported on the paper.

### Run the demo

In order to run the demo script on the sample images, simply run:

```Shell
python demo.py
```

If you want to use this in another project. Simply import the `models.RefineDetVGG16` class. Take a look at the file for clarification on the available parameters.

If you want to implement a different RefineDet based detector, you can extend `models.RefineDetBase` class and implement your own forward pass logic inside the `call` method. Follow the example in the `RefineDetVGG16` class.

If you want to retrain it on another dataset, I believe the files are easy enough to understant what is happening, so use the files `train_refinedet_voc.py` and `eval_refinedet_voc.py` as references for training and evaluation, respectively. If you are going to use the VGG16 CNN backbone [(Simonyan et al.)](https://arxiv.org/abs/1409.1556), download the pretrained weights from the [here](https://drive.google.com/open?id=1sBmajn6vOE7qJ8GnxUJt4fGPuffVUZox). These weights were provided by the popular [pierluigiferrari](https://github.com/pierluigiferrari)'s [ssd_keras](https://github.com/pierluigiferrari/ssd_keras) repository.

### Samples

Here are some sample detections on the PASCAL VOC 2007 test set.

<img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000001_det.png" width="430" height="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000002_det.png" width="430" height="430" />
<img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000004_det.png" width="430" height="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000006_det.png" width="430" height="430" />
<img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000008_det.png" width="430" height="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000010_det.png" width="430" height="430" />
<img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000011_det.png" width="430" height="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000013_det.png" width="430" height="430" />

### References

- **Original RefineDet Paper:** Zhang, Shifeng and Wen, Longyin and Bian, Xiao and Lei, Zhen and Li, Stan Z., *Single-Shot Refinement Neural Network for Object Detection*, CVPR 2018
- **Official Caffe implementation:** https://github.com/sfzhang15/RefineDet
- I also took some inspiration from some RefineDet implementations other than the official one, specially the one from [luuuyi](https://github.com/luuuyi). If you prefer Pytorch, check out [his implementation](https://github.com/luuuyi/RefineDet.PyTorch).
