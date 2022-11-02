import tensorflow as tf
import keras.layers
import keras.losses
import keras.optimizers
import keras.metrics


def conv_to_tensor(arg):
    """
    Takes any list or tuple or nparray and converts it to a tf.Tensor.
    For test purposes only.
    """
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg


class MyModel(keras.Model):

    def __init__(self):
        super().__init__()
        self.inputs = keras.layers.InputLayer(input_shape=(54,), name='Inputs')

        self.crossin = keras.layers.Dense(54, activation=tf.nn.relu, name='densecrossin')
        self.cross_lstm = keras.layers.CuDNNLSTM(324, return_sequences=True, name='lstmcross')
        self.crossout = keras.layers.Dense(18, activation=tf.nn.softmax, name='densecrossout')

        self.denseollin = keras.layers.Dense(54, activation=tf.nn.relu, name='denseollin')
        self.denseoll2 = keras.layers.Dense(400, activation=tf.nn.relu, name='denseoll2')
        self.denseoll0 = keras.layers.Dense(216, activation=tf.nn.softmax, name='denseoll0')

        self.densepllin = keras.layers.Dense(54, activation=tf.nn.relu, name='densepllin')
        self.densepll2 = keras.layers.Dense(680, activation=tf.nn.relu, name='densepll2')
        self.densepll3 = keras.layers.Dense(680, activation=tf.nn.sigmoid, name='densepll3')
        self.densepll0 = keras.layers.Dense(340, activation=tf.nn.softmax, name='densepll0')

    def call(self, inputs, training=None, mask=None):
        pass

    def pll_compile(self, name="PllCase"):
        self.model = keras.Sequential(name=name)
        self.model.add(self.inputs)
        self.model.add(self.densepllin)
        self.model.add(self.densepll2)
        self.model.add(self.densepll3)
        self.model.add(self.densepll0)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                           loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                           metrics=[keras.metrics.CategoricalAccuracy()]
                           )
        self.model.build()

    def oll_compile(self, name="OllCase"):
        self.model = keras.Sequential(name=name)
        self.model.add(self.inputs)
        self.model.add(self.denseollin)
        self.model.add(self.denseoll2)
        self.model.add(self.denseoll0)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                           loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                           metrics=[keras.metrics.CategoricalAccuracy()]
                           )
        self.model.build()
        # print(f"shape of output: {self.model.output_shape}")

    def f2l_compile(self, name="F2L"):
        self.model = keras.Sequential(name=name)
        pass

    def cross_compile(self, name="Cross"):
        self.model = keras.Sequential(name=name)
        self.model.add(self.inputs)
        self.model.add(self.crossin)
        self.model.add(self.cross_lstm)
        self.model.add(self.crossout)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                           loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=[keras.metrics.CategoricalAccuracy()]
                           )
        self.model.build()

    def __call__(self):
        self.model.summary()

    def train(self, trainingmatrix, trainingvector, callbacks=None, epochs=50, verbose=2):
        self.model.fit(
            trainingmatrix,
            trainingvector,
            callbacks=callbacks,
            epochs=epochs,
            verbose=verbose
        )

    def predict(self, predictmatrix, **kwargs):
        return self.model.predict(
            predictmatrix,
            kwargs
        )

    def eval(self, validationmatrix, validationvector):
        return self.model.evaluate(
            validationmatrix,
            validationvector,
        )

    def save(self, path, **kwargs):
        self.model.save_weights(filepath=path)

    def load(self, path):
        self.model.load_weights(filepath=path)


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
