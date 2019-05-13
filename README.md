# RefineDet implementation with TensorFlow

This project is intended to be a TensorFlow implementation of the RefineDet object detection model, from the paper *Single-Shot Refinement Neural Network for Object Detection* [(Zhang et al.)](https://arxiv.org/abs/1711.06897), originally made in Caffe.

The project is constructed mostly with the `tf.keras` [(Chollet et al.)](https://keras.io/) API specs and thought to work on the TensorFlow Eager Excecution mode.

I am conducting this as a side-project, so this repository is currently a work in progress. There are a few things that need to be better adjusted. In particular, I have not trained the model using the paper official parameters yet.

Nonetheless, preliminary tests have produced very good results, so I am releasing an inference script, as well as a weights file, trained on the 20 class PASCAL VOC 07+12 trainval datasets [(Everingham et al.)](http://host.robots.ox.ac.uk/pascal/VOC/). These weights are already achieving more the 70% mean AP on the VOC 2007 test set, not far from the ~80% reported on the paper.

### Inference

In order to make predictions, first download the weights file from [here](https://mega.nz/#!mi4HnIra!Cs5K54QUDFb9kSu5ciRzZoSe-AraCFxIH0rUjmmOPVQ) and put it inside the `weights` directory. Then, simply run:

```Shell
python refinedet_detection.py --images <IMAGES> --output_dir <OUTPUT>
```

`<IMAGES>` can be either the path to an image or to a directory containing the images to be fed to the model. `<OUTPUT>` is the directory in which to save the images with the drawn detected boxes. Run this script without arguments to apply the model to the sample images in the `samples` directory.

If you want to use this in another project. Simply import the `detect` method from `refinedet_detection.py`. Take a look at the file for clarification on the available parameters.

If you want to retrain it on another dataset, I believe the files are easy enough to understant what is happening, so use the files `train_refinedet_voc.py` and `voc_evaluation.py` as references for training and evaluation, respectively. If you are going to use the VGG16 CNN backbone [(Simonyan et al.)](https://arxiv.org/abs/1409.1556), download the pretrained weights from [here](https://mega.nz/#!zrg3BYZL!g2sZkmqjyvLZHXuENbcl6106N7ZbjiEG4tsu9_Qdqkg).

### Samples

Here are some sample detections on the PASCAL VOC 2007 test set. For now, the script does not print the class names, nor uses different box colors to discriminate them, as it is common on related projects.

<img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000001_det.png" width="430" height="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000002_det.png" width="430" height="430" />
<img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000004_det.png" width="430" height="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000006_det.png" width="430" height="430" />
<img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000008_det.png" width="430" height="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000010_det.png" width="430" height="430" />
<img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000011_det.png" width="430" height="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000013_det.png" width="430" height="430" />

### References

- **Original RefineDet Paper:** Zhang, Shifeng and Wen, Longyin and Bian, Xiao and Lei, Zhen and Li, Stan Z., *Single-Shot Refinement Neural Network for Object Detection*, CVPR 2018
- **Official Caffe implementation:** https://github.com/sfzhang15/RefineDet
- I also took some inspiration from some RefineDet implementations other than the official one, specially the one from [luuuyi](https://github.com/luuuyi). If you prefer Pytorch, check out [his implementation](https://github.com/luuuyi/RefineDet.PyTorch).
