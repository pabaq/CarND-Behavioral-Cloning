from keras import Sequential
from keras.layers import Lambda, Conv2D, BatchNormalization
from keras.layers import Flatten, Dense, Dropout

PROJECT_SHAPE = 160, 320, 3
CROPPED_SHAPE = 75, 320, 3
NVIDIA_SHAPE = 66, 200, 3


def get_network(network="nvidia", input_shape=PROJECT_SHAPE, dropout_rate=0.):
    """ Get the desired network.

    Args:
        network: up to now only the "nvidia" network is implemented
        input_shape: (height, width, channels) shape tuple of the input image
        dropout_rate: dropout probability for all implemented layers
    """

    if network.lower() == "nvidia":
        nvidia_model = Sequential()
        nvidia_model.add(Lambda(lambda x: x / 255. - 0.5,
                                input_shape=input_shape))
        nvidia_model.add(Conv2D(filters=24,
                                kernel_size=5,
                                strides=2,
                                padding='valid',
                                activation='relu'))
        # nvidia_model.add(BatchNormalization())
        nvidia_model.add(Conv2D(filters=36,
                                kernel_size=5,
                                strides=2,
                                padding='valid',
                                activation='relu'))
        # nvidia_model.add(BatchNormalization())
        nvidia_model.add(Conv2D(filters=48,
                                kernel_size=5,
                                strides=2,
                                padding='valid',
                                activation='relu'))
        # nvidia_model.add(BatchNormalization())
        nvidia_model.add(Conv2D(filters=64,
                                kernel_size=3,
                                strides=1,
                                padding='valid',
                                activation='relu'))
        # nvidia_model.add(BatchNormalization())
        nvidia_model.add(Conv2D(filters=64,
                                kernel_size=3,
                                strides=1,
                                padding='valid',
                                activation='relu'))
        # nvidia_model.add(BatchNormalization())
        nvidia_model.add(Flatten())
        nvidia_model.add(Dense(units=100,
                               activation='relu'))
        nvidia_model.add(Dropout(rate=dropout_rate))
        nvidia_model.add(Dense(units=50,
                               activation='relu'))
        nvidia_model.add(Dropout(rate=dropout_rate))
        nvidia_model.add(Dense(units=10,
                               activation='relu'))
        nvidia_model.add(Dense(units=1))

        return nvidia_model


if __name__ == "__main__":
    model = get_network("nvidia", NVIDIA_SHAPE)
    model.summary()
