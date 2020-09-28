import cv2
import numpy as np
import matplotlib.image as mpimg

from pathlib import Path

from model import *

CAMERA_STEERING_CORRECTION = 0.2


def image_path(sample, camera="center"):
    """ Transform the sample path to the repository structure.

    Args:
        sample: a sample (row) of the data dataframe. Usually drawn of a batch
         by the generator
        camera: the camera to extract the path for
    Returns:
        the converted image path string
    """
    return str(Path(f"./data/{sample[camera].split('data')[-1]}"))


def crop_image(image, top=60, bot=25):
    """ Crop the upper and lower borders of the given image.

    Args:
        image: the image to crop
        top: the pixels to crop from the upper part
        bot: the pixels to crop from the bottom part
    Returns:
        the cropped image
    """
    return image[top:-bot, :, :]


def resize_image(image, shape=NVIDIA_SHAPE[0:2]):
    """ Resize the image to shape.

    Args:
        image: input image
        shape: (height, width) tuple, defaults to Nvidia input shape (66, 200)
    Returns:
        the resized image
    """
    h, w = shape
    return cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_AREA)


def rgb2yuv(rgb_image):
    """ Convert the RGB image to YUV space. """
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)


def rgb2hsv(rgb_image):
    """ Convert the RGB image to HSV space. """
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)


def hsv2rgb(hsv_image):
    """ Convert the HSV image to RGB space. """
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)


def choose_camera(sample, camera='random', probs=None):
    """
    Choose an image for a specific camera and eventually adjust the steering.

    The steering of the left and right cameras is adjusted according to the
    defined constant CAMERA_STEERING_CONSTANT

    Args:
        sample: a sample (row) of the data dataframe. Usually drawn of a batch
         by the generator
        camera: 'random', 'left', 'center' or 'right'. If 'random' choose the
          camera with the given probabilities.
        probs: the probabilities to choose the left, center or right cameras. If
         None, the probabilities are uniform.
    Returns:
        a (image, steering) tuple
    """

    if camera == 'random':
        camera = np.random.choice(["left", "center", "right"], p=probs)
    image = mpimg.imread(image_path(sample, camera=camera))
    steering = sample["steer"]
    if camera == "left":
        steering += CAMERA_STEERING_CORRECTION
    elif camera == "right":
        steering -= CAMERA_STEERING_CORRECTION
    return image, steering


def flip(image, steering, prob=0.5):
    """ Flip the image and steering with the given probability.

    Args:
        image: the image to flip
        steering: the steering corresponding to the image
        prob: the flip probability
    Returns:
        the augmented image
    """
    if np.random.random() < prob:
        image = cv2.flip(image, 1)
        steering *= -1
    return image, steering


def shadow(rgb_image, prob=0.5):
    """ Add a shadow to the rgb image with the given probability.

    The shadow is created by converting the RGB image into HSV space and
    modifying the value channel in a random range. The area in which the value
    is modified is defined by a convex hull created for 6 randomly chosen points
    in the lower half of the image.

    Args:
        rgb_image: the image to add the shadow to. Has to be in RGB space.
        prob: the probability to add the shadow
    Returns:
        the augmented image
    """

    if np.random.random() < prob:
        width, height = rgb_image.shape[1], rgb_image.shape[0]
        # Get 6 random vertices in the lower half of the image
        x = np.random.randint(-0.1 * width, 1.1 * width, 6)
        y = np.random.randint(height * 0.5, 1.1 * height, 6)
        vertices = np.column_stack((x, y)).astype(np.int32)
        vertices = cv2.convexHull(vertices).squeeze()
        # Intilialize mask
        mask = np.zeros((height, width), dtype=np.int32)
        # Create the polygon mask
        cv2.fillPoly(mask, [vertices], 1)
        # Adjust value
        hsv = rgb2hsv(rgb_image)
        v = hsv[:, :, 2]
        hsv[:, :, 2] = np.where(mask, v * np.random.uniform(0.5, 0.8), v)
        rgb_image = hsv2rgb(hsv)
    return rgb_image


def brightness(rgb_image, low=0.6, high=1.4, prob=0.5):
    """ Modify the brighntess of the rgb image with the given probability.

    The brightness is modified by converting the RGB image into HSV space and
    adusting the value channel in a random range between the low and high
    bounds.

    Args:
        rgb_image: the image to modify the brightness. Has to be in RGB space.
        low: lower value bound
        high: upper value bound
        prob: the probability to modify the brightness
    Returns:
        the augmented image
    """

    if np.random.random() < prob:
        hsv = rgb2hsv(rgb_image)
        value = hsv[:, :, 2]
        hsv[:, :, 2] = np.clip(value * np.random.uniform(low, high), 0, 255)
        rgb_image = hsv2rgb(hsv)
    return rgb_image


def shift(image, steering, shiftx=60, shifty=20, prob=0.5):
    """ Shift the image and adjust the steering with the given probability.

    The steering of the shifted image is adjusted depending on the amount of
    pixels shifted in the width direction.

    Args:
        image: the image to shift.
        steering: the corresponding steering.
        shiftx: the upper bound of pixels to shift in the width direction
        shifty: the upper bound of pixels to shift in the height direction
        prob: the probability to shift the image
    Returns:
        the augmented image
    """

    if np.random.random() < prob:
        # The angle correction per pixel is derived from the angle correction
        # specified for the side cameras. It is estimated that the images of two
        # adjacent cameras are shifted by 80 pixels (at the bottom of the image)
        angle_correction_per_pixel = CAMERA_STEERING_CORRECTION / 80
        # Draw translations in x and y directions from a uniform distribution
        tx = int(np.random.uniform(-shiftx, shiftx))
        ty = int(np.random.uniform(-shifty, shifty))
        # Transformation matrix
        mat = np.float32([[1, 0, tx],
                          [0, 1, ty]])
        # Transform image and correct steering angle
        height, width, _ = image.shape
        image = cv2.warpAffine(image, mat, (width, height),
                               borderMode=cv2.BORDER_REPLICATE)
        steering += tx * angle_correction_per_pixel
    return image, steering


def augment(sample, camera_probs=None, flip_prob=0.5, shadow_prob=0.5,
            bright_prob=0.5, shift_prob=0.5, ):
    """ Augment the sample with the given probabilities.

    Args:
        sample: a sample (row) of the data dataframe. Usually drawn of a batch
         by the generator
        camera_probs: the probabilities to draw left, center or right camera
         images
        flip_prob: probability for an image to be flipped
        shadow_prob: probability of shadow additon to the image
        bright_prob: probability to modify the brightness of the image
        shift_prob: probability for and image to be shifed
    """
    image, steering = choose_camera(sample, probs=camera_probs)
    image, steering = flip(image, steering, prob=flip_prob)
    image = shadow(image, prob=shadow_prob)
    image = brightness(image, prob=bright_prob)
    image, steering = shift(image, steering, prob=shift_prob)
    return image, steering
