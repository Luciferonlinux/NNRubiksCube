from keras.models import Sequential
from keras.layers import Embedding, Conv3D
from keras.callbacks import ModelCheckpoint
import os


def create_model():
    """
    Builds the NN
    :return: the NN
    """
    embedding_in = 6
    embedding_out = 27

    model = Sequential()

    model.add(
        [
            Embedding(
                input_dim=embedding_in,
                output_dim=embedding_out
            ),
            Conv3D(

            )
        ]
    )

    model.compile('adamax', 'mse')
    model.summary()

    return model


def save_checkpoint(checkpoint_path="NNTraining/cp.ckpt"):
    """
    Saves the models Weights
    :param checkpoint_path: defaults to NNTraining/cp.ckpt
    :return:
    """
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )
    return cp_callback


# create model
model = create_model()

# show the models architecture and details
model.summary()
