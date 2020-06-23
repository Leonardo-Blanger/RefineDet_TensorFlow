import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D


class VGG16ReducedFC(tf.keras.Model):
    def __init__(self, output_features=['conv7'],
                 kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                 **kwargs):
        
        super(VGG16ReducedFC, self).__init__(**kwargs)
        self.output_features = output_features

        def _get_conv(filters, name):
            return Conv2D(filters=filters, kernel_size=3, padding='same',
                          activation='relu', name=name,
                          kernel_regularizer=kernel_regularizer)

        self.channel_means = tf.constant([123, 117, 104], tf.float32)

        self.conv1_1 = _get_conv(64, name='conv1_1')
        self.conv1_2 = _get_conv(64, name='conv1_2')
        self.pool1   = MaxPool2D(pool_size=2, name='pool1')

        self.conv2_1 = _get_conv(128, name='conv2_1')
        self.conv2_2 = _get_conv(128, name='conv2_2')
        self.pool2   = MaxPool2D(pool_size=2, name='pool2')

        self.conv3_1 = _get_conv(256, name='conv3_1')
        self.conv3_2 = _get_conv(256, name='conv3_2')
        self.conv3_3 = _get_conv(256, name='conv3_3')
        self.pool3   = MaxPool2D(pool_size=2, name='pool3')

        self.conv4_1 = _get_conv(512, name='conv4_1')
        self.conv4_2 = _get_conv(512, name='conv4_2')
        self.conv4_3 = _get_conv(512, name='conv4_3')
        self.pool4   = MaxPool2D(pool_size=2, name='pool4')

        self.conv5_1 = _get_conv(512, name='conv5_1')
        self.conv5_2 = _get_conv(512, name='conv5_2')
        self.conv5_3 = _get_conv(512, name='conv5_3')
        self.pool5   = MaxPool2D(pool_size=2, name='pool5')

        self.conv6 = Conv2D(filters=1024, kernel_size=3, dilation_rate=3,
                            padding='same', activation='relu', name='fc6',
                            kernel_regularizer=kernel_regularizer)
        self.conv7 = Conv2D(filters=1024, kernel_size=1, name='fc7', activation='relu',
                            kernel_regularizer=kernel_regularizer)

        self.all_layers = [
            self.conv1_1, self.conv1_2, self.pool1,
            self.conv2_1, self.conv2_2, self.pool2,
            self.conv3_1, self.conv3_2, self.conv3_3, self.pool3,
            self.conv4_1, self.conv4_2, self.conv4_3, self.pool4,
            self.conv5_1, self.conv5_2, self.conv5_3, self.pool5,
            self.conv6, self.conv7,
        ]

    @tf.function
    def call(self, x):
        x -= self.channel_means
        output = []
        for layer in self.all_layers:
            x = layer(x)
            if layer.name in self.output_features:
                output.append(x)
        return output
