import tensorflow as tf
import keras.layers

from random import randint
cubelist = [randint(0, 5) for _ in range(54)]


def conv_to_tensor(arg):
    """
    Takes any list or tuple or nparray and converts it to a tf.Tensor
    """
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg


print(tf.config.list_physical_devices())


class MyModel(keras.Model):

    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = keras.layers.Dense(5, activation=tf.nn.softmax)
        self.dropout = keras.layers.Dropout(0.5)

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        return self.dense2

