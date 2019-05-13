import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPool2D

class VGG16ReducedFC(tf.keras.Model):
    def __init__(self, features_to_return = ['conv7']):
        super(VGG16ReducedFC, self).__init__(name = 'VGG16ReducedFC')

        self.features_to_return = features_to_return

        self.conv1_1 = Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'conv1_1')
        self.conv1_2 = Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'conv1_2')
        self.pool1   = MaxPool2D(pool_size = (2, 2), name = 'pool1')

        self.conv2_1 = Conv2D(128, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'conv2_1')
        self.conv2_2 = Conv2D(128, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'conv2_2')
        self.pool2   = MaxPool2D(pool_size = (2, 2), name = 'pool2')

        self.conv3_1 = Conv2D(256, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'conv3_1')
        self.conv3_2 = Conv2D(256, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'conv3_2')
        self.conv3_3 = Conv2D(256, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'conv3_3')
        self.pool3   = MaxPool2D(pool_size = (2, 2), name = 'pool3')

        self.conv4_1 = Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'conv4_1')
        self.conv4_2 = Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'conv4_2')
        self.conv4_3 = Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'conv4_3')
        self.pool4   = MaxPool2D(pool_size = (2, 2), name = 'pool4')

        self.conv5_1 = Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'conv5_1')
        self.conv5_2 = Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'conv5_2')
        self.conv5_3 = Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu', name = 'conv5_3')
        self.pool5   = MaxPool2D(pool_size = (2, 2), name = 'pool5')

        self.conv6 = Conv2D(filters = 1024, kernel_size = (3, 3), dilation_rate = (3, 3), padding = 'same', activation = 'relu', name = 'conv6')
        self.conv7 = Conv2D(filters = 1024, kernel_size = (1, 1), activation = 'relu', name = 'conv7')

    def call(self, x):
        layers = [
            self.conv1_1, self.conv1_2, self.pool1,
            self.conv2_1, self.conv2_2, self.pool2,
            self.conv3_1, self.conv3_2, self.conv3_3, self.pool3,
            self.conv4_1, self.conv4_2, self.conv4_3, self.pool4,
            self.conv5_1, self.conv5_2, self.conv5_3, self.pool5,
            self.conv6, self.conv7
        ]

        output = []
        for layer in layers:
            x = layer(x)
            if layer.name in self.features_to_return:
                output.append(x)

        return output

        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.conv7(x)

        return x
