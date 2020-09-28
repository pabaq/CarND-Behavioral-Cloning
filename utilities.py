import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle

from augmentation import *

plt.style.use('seaborn')
sns.set_color_codes()


def load_data(dirs):
    """ Load image paths and corresponding steering angles into a dataframe.

    Args:
        dirs: list of directories to get the data from
    Returns:
        a dataframe with image path columns "center", "left", "right" and
         steering angle column "steer"
    """

    names = ["center", "left", "right", "steer", "throttle", "break", "speed"]
    df_list = []
    for directory in dirs:
        df_list.append(pd.read_csv(f"{directory}/driving_log.csv", names=names))
    df = pd.concat(df_list, ignore_index=True)
    return df[["center", "left", "right", "steer"]]


def draw_batches(data, batch_size=128):
    """ Create a list of batches for the given data.

    Args:
        data: the dataframe returned by load_data
        batch_size: number of samples to include in each batch
    Returns:
        a list of batches. Each batch is a part of the data dataframe with
         batch_size rows.
    """

    minibatches = []
    num_samples = data.shape[0]

    # Complete mini batches
    complete_batches = num_samples // batch_size
    for i in range(0, complete_batches):
        minibatch = data.iloc[i * batch_size: i * batch_size + batch_size]
        minibatches.append(minibatch)

    # Eventually uncomplete last minibatch
    if num_samples % batch_size != 0:
        minibatch = data.iloc[complete_batches * batch_size: num_samples]
        minibatches.append(minibatch)

    return minibatches


def plot_histogram(generator, num_samples, name="histogram",
                   camera_probs=None, flip_prob=0.5, shift_prob=0.5):
    """ Create a histogram of the data produced by the given generator.

    Args:
        generator: the initialized data generator
        num_samples: number of samples to generate for the histogram
        name: the name of the output image
        camera_probs: the probabilities to draw left, center or right camera
         samples
        flip_prob: probability for an sample to be flipped
        shift_prob: probability for and sample to be shifted
    """

    counter = 0
    angles = []
    # Draw samples until the desired number is collected
    while counter < num_samples:
        _, steering_angles = next(generator)
        counter += len(steering_angles)
        angles.append(steering_angles)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    ax.hist(np.concatenate(angles).tolist(), bins=20, color="b")
    ax.set_title(f"Augmented Data ({counter} samples)\n"
                 f"Probabilities:   "
                 f"camera = [{camera_probs[0]:.2f}, "
                 f"{camera_probs[1]:.2f}, "
                 f"{camera_probs[2]:.2f}]   "
                 f"flip = {flip_prob:.1f}   "
                 f"shift = {shift_prob:.1f}",
                 size=14)
    ax.set_xlim([-1, 1])
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("steering", size=14)
    ax.set_ylabel("samples", size=14)
    fig.savefig(name)


def plot_augmentation(samples, name="augmentation",
                      camera_probs=None, flip_prob=1.0, shadow_prob=1.0,
                      bright_prob=1.0, shift_prob=1.0,
                      crop=False, resize=False, yuv=False):
    """
    Visualization of the augmentation and preprocessing of the given samples.

    Subplots are only created for augmentations with probalities != 0 and True
    preprocessing steps.

    Args:
        samples: samples dataframe
        name: output name for the created plot
        camera_probs: the probabilities to draw left, center or right camera
         samples
        flip_prob: probability for an sample to be flipped
        shadow_prob: probability of shadow creation in a sample
        bright_prob: probability to modify the brightness of a sample
        shift_prob: probability for and sample to be shifted
        crop: if True, crop the sample image
        resize: if True, resize the sample image to Nvidia shape
        yuv: if True, convert the sample image from RGB to YUV space
    """

    pars = [flip_prob, shadow_prob, bright_prob, shift_prob, crop, resize, yuv]
    rows = sum([1 for par in pars if par]) + 1
    cols = samples.shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2), dpi=100)
    ylabels = ["original", "flip", "shadow",
               "brightness", "shift", "crop", "resize", "YUV"]

    for col, (_, sample) in enumerate(samples.iterrows()):

        # Choose camera
        try:
            ax = axes[0, col]
        except IndexError:
            ax = axes[col]
        image, steering = choose_camera(sample,
                                        camera='random',
                                        probs=camera_probs)
        ax.imshow(image)
        if col == 0:
            ax.set_ylabel(f"original", size=14)
            ax.set_title(f"steering = {steering:.3f}", size=14)
        else:
            ax.set_title(f"{steering:.3f}", size=14)

        flip_flag = flip_prob
        shadow_flag = shadow_prob
        bright_flag = bright_prob
        shift_flag = shift_prob
        crop_flag = crop
        resize_flag = resize
        yuv_flag = yuv
        for row in range(1, rows):
            ax = axes[row, col]
            if flip_prob and flip_flag:
                # Flip image
                image, steering = flip(image, steering, prob=flip_prob)
                flip_flag = False
            elif shadow_prob and shadow_flag:
                # Add shadow
                image = shadow(image, prob=shadow_prob)
                shadow_flag = False
            elif bright_prob and bright_flag:
                # Change brightness
                image = brightness(image, prob=bright_prob)
                bright_flag = False
            elif shift_prob and shift_flag:
                # Shift image
                image, steering = shift(image, steering, prob=shift_prob)
                shift_flag = False
            elif crop and crop_flag:
                # Crop image
                image = crop_image(image)
                crop_flag = False
            elif resize and resize_flag:
                # Resize image
                image = resize_image(image)
                resize_flag = False
            elif yuv and yuv_flag:
                # Resize image
                image = rgb2yuv(image)
                yuv_flag = False

            ax.imshow(image)
            if col == 0:
                ax.set_ylabel(f"{ylabels[row]}", size=14)
            ax.set_title(f"{steering:.3f}", size=14)

    for ax in axes.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(name)


def plot_history(history, name="history"):
    """ Create a plot of the training history.

    Args:
        history: the keras training history object
        name: name of the ouput file
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(loss) + 1), loss, label="training set")
    ax.plot(np.arange(1, len(val_loss) + 1), val_loss, label="validation set")
    ax.set_title('model mean squared error loss')
    ax.set_ylabel('mean squared error loss')
    ax.set_xlabel('epoch')
    ax.legend(loc='upper right')
    fig.savefig(name)


def generator(data, batch_size=128, training=True,
              camera_probs=None, flip_prob=0.5, shadow_prob=0.5,
              bright_prob=0.5, shift_prob=0.5,
              crop=False, resize=False, yuv=False):
    """ Generator to produce sample batches for the training and validation.

    The amount of augmentations can be controlled by their probabilites.
    Preprocessing steps are only performed if explicitly activated.

    To get the validation batches set the training parameter to False. In that
    case no augmentation is perfomed, and all samples are drawn for the center
    camera. Preprocessing steps are performed if activated.

    Args:
        data: the training or validation dataframe
        batch_size: the number of samples per batch
        training: if False, the generator will delivier validation batches
        camera_probs: the probabilities to draw left, center or right camera
         samples
        flip_prob: probability for an sample to be flipped
        shadow_prob: probability of shadow creation in a sample
        bright_prob: probability to modify the brightness of a sample
        shift_prob: probability for and sample to be shifted
        crop: if True, crop the sample image
        resize: if True, resize the sample image to Nvidia shape
        yuv: if True, convert the sample image from RGB to YUV space
    """

    # Infinity loop. The generator will produce batches as long as needed
    while True:
        shuffled_data = shuffle(data)
        batches = draw_batches(shuffled_data, batch_size)

        for batch in batches:
            images = []
            steers = []
            for _, sample in batch.iterrows():
                if training is True:
                    image, steering_angle = augment(sample,
                                                    camera_probs=camera_probs,
                                                    flip_prob=flip_prob,
                                                    shadow_prob=shadow_prob,
                                                    bright_prob=bright_prob,
                                                    shift_prob=shift_prob)
                    if crop is True:
                        image = crop_image(image)
                    if resize is True:
                        image = resize_image(image)
                    if yuv is True:
                        image = rgb2yuv(image)
                else:
                    # For validation no augmentation is performed. All samples
                    # are of the center camera. Only the activated preprocessing
                    # steps are perfomed.
                    image, steering_angle = choose_camera(sample,
                                                          camera="center")
                    if crop is True:
                        image = crop_image(image)
                    if resize is True:
                        image = resize_image(image)
                    if yuv is True:
                        image = rgb2yuv(image)

                images.append(image)
                steers.append(steering_angle)

            yield np.array(images), np.array(steers)
