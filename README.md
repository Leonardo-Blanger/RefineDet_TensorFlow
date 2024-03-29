# RefineDet implementation with TensorFlow

This project is intended to be a TensorFlow implementation of the RefineDet object detection architecture, from the paper *Single-Shot Refinement Neural Network for Object Detection* [(Zhang et al.)](https://arxiv.org/abs/1711.06897), originally made in Caffe.

I mostly worked on this project during spare time, and as a way to learn more about TensorFlow in practice. Slight details might deviate from the official Caffe implementation, but all the main ideas are present.

The code has been completely adapted to work with TensorFlow 2.0, and it only uses TF operations, instead of python logic + `tf.py_function` as was the case in the initial version of this project, which gave a significant boost in training speed.

I am releasing an inference demo script, as well as a weights file, trained on the 20 class PASCAL VOC 07+12 trainval datasets [(Everingham et al.)](http://host.robots.ox.ac.uk/pascal/VOC/), with 320 x 320 input size. These weights are  achieving around 80% mean AP on the VOC 2007 test set, which is about the same as reported on the paper. But note that I used my own custom Python implementation of the VOC Mean Average Precision, which might deviate slightly from the official Matlab version.

### Requirements

You can create a conda environment with all necessary dependencies.

``` Shell
# inside the repo's root directory
conda env create --file=env.yml --prefix=./env
conda activate ./env
```

### Running the demo

In order to run the demo script on the sample images, first download the pretrained weights file from [here](https://mega.nz/file/37ZVSIZb#egkUaB0RhJ6FVYUMKp1WYOrMeSvYeQAIPBMm3VOhDTw) into `./weights`, and then simply run:

```Shell
python demo.py
```

and it will perform inference on a few Pascal VOC test images inside the `samples` directory. The results will be displayed on the screen and saved inside `samples/detections`.

### Running evaluation

If you want to perform evaluation on the whole Pascal VOC 2007 test set, as done in the paper, first download the 2007 test partition of the dataset, and extract it inside `./data` . You can find the the main Pascal VOC files in the [pjreddie mirror](https://pjreddie.com/projects/pascal-voc-dataset-mirror/).

Assuming you already have the weights file inside `./weights`, you just have to run.

```Shell
python eval_refinedet_voc.py
```

Take a look in this file to see optional parameters.

### Training

If you want to train from scratch, you will need to download the pretrained weigths for the VGG16 backbone CNN from [here](https://drive.google.com/open?id=1sBmajn6vOE7qJ8GnxUJt4fGPuffVUZox). These weights were provided by the popular [pierluigiferrari](https://github.com/pierluigiferrari)'s [ssd_keras](https://github.com/pierluigiferrari/ssd_keras) repository.

You will also need the Pascal VOC 2007 and 2012 trainval partitions, which you can also find in the [pjreddie mirror](https://pjreddie.com/projects/pascal-voc-dataset-mirror/).

If you have the pretrained weights inside `./weights` and the VOC data extracted inside `./data`, then training is just running the train script:

```Shell
python train_refinedet_voc.py
```

Take a look in this file to see optional parameters.

### Using this project

You can install the main package as follows. If you are using an isolated environment (see above), make sure it is active.

```
# inside the repo's root directory
python -m pip install .
```

If you want to use this implementation in another project, simply import the `refinedet_tensorflow.models.RefineDetVGG16` class. Take a look at the file for clarification on the available parameters.

If you want to implement a different RefineDet based detector, you can extend `refinedet_tensorflow.models.RefineDetBase` class and implement your own forward pass logic inside the `call` method. Follow the example in the `RefineDetVGG16` class.

If you want to retrain it on another dataset, I believe the files are easy enough to understant what is happening, so use the scripts `train_refinedet_voc.py` and `eval_refinedet_voc.py` as references for training and evaluation, respectively. If you are going to use the VGG16 CNN backbone [(Simonyan et al.)](https://arxiv.org/abs/1409.1556), download the pretrained weights mentioned above into `./weights`.

### Samples

Here are some sample detections on the PASCAL VOC 2007 test set.

<img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000001_det.jpg" width="430" height="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000002_det.jpg" width="430" height="430" />
<img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000004_det.jpg" width="430" height="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000022_det.jpg" width="430" height="430" />
<img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000008_det.jpg" width="430" height="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000010_det.jpg" width="430" height="430" />
<img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000011_det.jpg" width="430" height="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/RefineDet_TensorFlow/master/samples/detections/000038_det.jpg" width="430" height="430" />

### References

- **Original RefineDet Paper:** Zhang, Shifeng and Wen, Longyin and Bian, Xiao and Lei, Zhen and Li, Stan Z., *Single-Shot Refinement Neural Network for Object Detection*, CVPR 2018
- **Official Caffe implementation:** https://github.com/sfzhang15/RefineDet
- I also took some inspiration from some RefineDet implementations other than the official one, specially the one from [luuuyi](https://github.com/luuuyi). If you prefer Pytorch, check out [his implementation](https://github.com/luuuyi/RefineDet.PyTorch).
