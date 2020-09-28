from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from itertools import product

from utilities import *
from model import *

# Definition of the pipeline parameters
# Directory to store the pipelines output
save_dir = "models/V16"
# Definition of the data to use
tracks = ["lake_track", "jungle_track"]
maneuvers = ["standard", "recovery", "reverse"]
# Training parameters
batch_size = 128
epochs = 10
dropout_rate = 0
augmentation_kwargs = dict(
    camera_probs=[0.4, 0.2, 0.4],  # left, center, right
    flip_prob=0.5,
    shadow_prob=0.5,
    bright_prob=0.5,
    shift_prob=0.5)
preprocessing_kwargs = dict(
    crop=True,  # if True, crop upper and lower sections of the image
    resize=True,  # if True, resize to Nvidia input shape (66, 200)
    yuv=True)  # if True, convert RGB to YUV space


if __name__ == "__main__":

    # Get the network
    if preprocessing_kwargs["crop"] and not preprocessing_kwargs["resize"]:
        input_shape = CROPPED_SHAPE
    elif preprocessing_kwargs["resize"]:
        input_shape = NVIDIA_SHAPE
    else:
        input_shape = PROJECT_SHAPE
    network = get_network(input_shape=input_shape, dropout_rate=dropout_rate)

    # Load traning and validation data
    directories = [f"./data/{t}/{m}" for t, m in product(tracks, maneuvers)]
    data = load_data(directories)
    train_data, valid_data = train_test_split(data, test_size=0.2)

    # Evenutally create an example augmentation plot for the README
    plot_augmentation(train_data.iloc[0:5], crop=True)

    # Create the pipeline informational plots
    flag = False
    if flag is True:
        # Save pipeline example images
        plot_augmentation(train_data.iloc[0:5],
                          name=f"{save_dir}/augmentation",
                          **augmentation_kwargs,
                          **preprocessing_kwargs)
        # Save pipeline histogram
        data_generator = generator(data,
                                   batch_size=batch_size,
                                   training=True,
                                   **augmentation_kwargs,
                                   **preprocessing_kwargs)
        plot_histogram(data_generator,
                       num_samples=data.shape[0],
                       name=f"{save_dir}/data_histogram",
                       camera_probs=augmentation_kwargs["camera_probs"],
                       flip_prob=augmentation_kwargs["flip_prob"],
                       shift_prob=augmentation_kwargs["shift_prob"])

    # Train pipeline and save the trained model
    flag = False
    if flag is True:
        train_generator = generator(train_data,
                                    batch_size=batch_size,
                                    training=True,
                                    **augmentation_kwargs,
                                    **preprocessing_kwargs)
        valid_generator = generator(valid_data,
                                    batch_size=batch_size,
                                    training=False,
                                    **preprocessing_kwargs)

        # Callbacks
        checkpoint = ModelCheckpoint(save_dir + '/nvidia_model_{epoch:03d}.h5',
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=False,
                                     mode='auto')

        network.compile(loss='mse', optimizer=Adam(lr=0.001))
        history = network.fit_generator(
            generator=train_generator,
            steps_per_epoch=len(train_data) // batch_size,
            validation_data=valid_generator,
            validation_steps=len(valid_data) // batch_size,
            epochs=epochs, verbose=2, callbacks=[checkpoint])

        plot_history(history, name=f"{save_dir}/history.png")
