import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K


class L2Norm(tf.keras.layers.Layer):
    def __init__(self, initial_scale=1.0, **kwargs):
        super(L2Norm, self).__init__(**kwargs)
        self.initial_scale = initial_scale
    
    def build(self, input_shape):
        self.scale = self.add_weight(name=self.name+'_scale',
                                     shape=[1, 1, 1, input_shape[-1]],
                                     initializer=Constant(self.initial_scale),
                                     trainable=True)
        super(L2Norm, self).build(input_shape)

    def call(self, x):
        return K.l2_normalize(x, axis=-1) * self.scale

    def compute_output_shape(self, input_shape):
        return input_shape
        
