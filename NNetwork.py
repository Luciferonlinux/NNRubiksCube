import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Conv3D, Input


def convtotensor(arg):
    """
    Takes any list or tuple or nparray and converts it to a tf.Tensor
    """
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg


def NN(cubelist):
    """
    Takes the Cube as List and runs it through the Neural network
    """

    cubetensor = convtotensor(cubelist)
    shape = cubetensor.shape
    return shape
    # model = Sequential()
    # model.add(
    #    Embedding(
    #        input_dim=cubetensor.tf.tensor
    #
    #    )
    # )


NN(convtotensor([1, 1, 1, 2, 2, 3, 5, 1, 2, 4, 2, 3, 4, 1]))
