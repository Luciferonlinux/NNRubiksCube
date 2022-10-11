import tensorflow as tf
import keras.layers
import keras.losses


def conv_to_tensor(arg):
    """
    Takes any list or tuple or nparray and converts it to a tf.Tensor
    """
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg


class MyModel(keras.Model):

    def __init__(self):
        super().__init__()
        self.inputs = keras.layers.InputLayer(input_shape=(54,), name='Inputs')
        self.denseoll1 = keras.layers.Dense(75, activation=tf.nn.sigmoid, name='denseoll1')
        self.denseoll2 = keras.layers.Dense(150, activation=tf.nn.sigmoid, name='denseoll2')
        self.denseoll3 = keras.layers.Dense(216, activation=tf.nn.softmax, name='denseoll3')

    def call(self, inputs, training=None, mask=None):
        pass

    def pll_compile(self, name="PllCase"):
        self.model = keras.Sequential(name=name)
        self.model.add(self.embedding)
        self.model.add(self.densepll1)
        self.model.compile(optimizer="adamax", loss=keras.losses.BinaryCrossentropy())

    def oll_compile(self, name="OllCase"):
        self.model = keras.Sequential(name=name)
        self.model.add(self.inputs)
        print(f"shape of input: {self.model.output_shape}")
        self.model.add(self.denseoll1)
        print(f"shape of dense out: {self.model.output_shape}")
        self.model.add(self.denseoll2)
        print(f"shape of dense out: {self.model.output_shape}")
        self.model.add(self.denseoll3)
        print(f"shape of dense out: {self.model.output_shape}")
        self.model.compile(optimizer="adamax", loss=keras.losses.BinaryCrossentropy(from_logits=False))
        print(f"shape of output: {self.model.output_shape}")

    def f2l_compile(self, name="F2L"):
        self.model = keras.Sequential(name=name)
        pass

    def cross_compile(self, name="Cross"):
        self.model = keras.Sequential(name=name)
        pass

    def __call__(self):
        self.model.summary()

    def train(self, trainingmatrix, trainingvector):
        self.model.fit(
            trainingmatrix,
            trainingvector,
            epochs=10
        )

    def predict(self, predictmatrix):
        return self.model.predict(
            predictmatrix
        )


if __name__ == '__main__':
    from random import randint
    cubelist = [randint(0, 5) for _ in range(54)]

    inputvector = conv_to_tensor(cubelist)
    resultvector = conv_to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                                   3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    model = MyModel()
    model.oll_compile()
    model()
    model.train(inputvector, resultvector)
    model.predict(cubelist)
