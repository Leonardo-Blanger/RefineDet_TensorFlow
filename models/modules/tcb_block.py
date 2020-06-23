import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose


class TCBBlock(tf.keras.Model):
    def __init__(self,
                 kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                 **kwargs):
        super(TCBBlock, self).__init__(**kwargs)

        self.conv1 = Conv2D(256, kernel_size=3,
                            padding='same', activation='relu',
                            kernel_regularizer=kernel_regularizer)
        
        self.conv2 = Conv2D(256, kernel_size=3,
                            padding='same', activation=None,
                            kernel_regularizer=kernel_regularizer)

        self.tconv = Conv2DTranspose(256, kernel_size=2, strides=2,
                                     padding='same', activation=None,
                                     kernel_regularizer=kernel_regularizer)

        self.conv3 = Conv2D(256, kernel_size=3,
                            padding='same', activation='relu',
                            kernel_regularizer=kernel_regularizer)


    @tf.function
    def call(self, ftm, next_ftm=None):
        x = self.conv1(ftm)
        x = self.conv2(x)

        if next_ftm is not None:
            x += self.tconv(next_ftm)

        x = tf.nn.relu(x)
        x = self.conv3(x)
        return x
