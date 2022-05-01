This project is part of Udacity's [Self-Driving-Car Nanodegree][Course]. The project
resources and build instructions can be found [here][Project], the required 
simulator [here][UdacitySimulator] (Term 1).

## End-to-end deep learning

The goal of this project is to teach a Convolutional Neural Network to mimic human driving
behaviour by providing only the car's front camera images and the corresponding steering 
commands. The network will learn to predict the correct steering, when fed with unseen road
images. 

The approach is comparable to Nvidias [End-to-End Deep Learning for Self-Driving Cars][Nvidia2016] 
and we will therefore adopt their CNN for this project. You can find their papers in the 
[References](#references) in the end. Instead of using real-world data we will collect the training data
by driving a few laps in Udacity's [Self-Driving Car Simulator][UdacitySimulator]. Additionally,
we will augment the collected data to help the network generalize and recover from adverse 
positions.

The complete code of this project can be found in the following modules:

- [``model.py``][model]: Implementation of Nvidia's network with Keras
- [``utilities.py``][utilities]: Utilities for the loading and visualization of the data. 
  Furthermore, definition of the generator to feed the network with the training
  and validation batches.
- [``augmenation.py``][augmentation]: Augmentation and preprocessing functions used to extend
  and preprocess the collected data before it is fed into the network.
- [``Behavioural_Cloning.py``][main]: Setting all relevant parameters and training of the network.
- [``drive.py``][drive]: Needed to let a trained network drive the car in autonomous mode 
  in the simulator.

Several examinations and parameter variations were performed. The trained models for each 
investigation are saved as ``*.h5`` files in the subdirectories of the [``models``][models]
folder.

To have a model drive the car autonomously, a track must be started in the simulator in 
autonomous mode, and the following command must be entered into the console 
``python drive.py model.h5``. Note, that the [``preprocessing_kwargs``][prep] in 
[``Behavioural_Cloning.py``][main] must be set as they were set during the training process 
of the model.

## Network Architecture
![][cnn]  
&nbsp;

Nvidia's CNN takes an image of the shape (66, 200) in the YUV color space as input. A normalization 
layer is followed by 5 convolutional layers, of which the first three have a ``5x5`` kernel 
with a ``2x2`` stride and the last two a non-strided ``3x3`` kernel. The output of the fifth
convolutional layer is flattened and is followed by three fully connected layers of the specified 
size. The output of the network is a single node containing the steering value to be learned by 
regression.

In the network's implementation for this project ``Relu`` activations are used throughout all 
layers. A dropout layer is implemented after each dense layer. The dropout rate can be adjusted 
individually for each study. However, overfitting was not really a problem in this project. 
The dropout layers were kept inactive for most of the examinations. The network is 
implemented using Keras as shown below.

````python
nvidia_model = Sequential()
nvidia_model.add(Lambda(lambda x: x / 255. - 0.5,
                        input_shape=input_shape))
nvidia_model.add(Conv2D(filters=24,
                        kernel_size=5,
                        strides=2,
                        padding='valid',
                        activation='relu'))
nvidia_model.add(Conv2D(filters=36,
                        kernel_size=5,
                        strides=2,
                        padding='valid',
                        activation='relu'))
nvidia_model.add(Conv2D(filters=48,
                        kernel_size=5,
                        strides=2,
                        padding='valid',
                        activation='relu'))
nvidia_model.add(Conv2D(filters=64,
                        kernel_size=3,
                        strides=1,
                        padding='valid',
                        activation='relu'))
nvidia_model.add(Conv2D(filters=64,
                        kernel_size=3,
                        strides=1,
                        padding='valid',
                        activation='relu'))
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
```` 

## Data Collection
![][simulator]

The simulator has two tracks, which we call lake and jungle. Although the project does 
not require us to teach the network to drive the more challenging jungle track, we will 
attempt to train the network to master both tracks. We collect the following data on each 
of both tracks:

- 2 laps of steady driving at the lane center
- 1 lap of driving counterclockwise
- 1 lap of recovery driving

In the recovery lap, the vehicle is moved several times to an adverse position and the 
situation is subsequently recovered by driving the car back to the lane center. During the 
drive, the steering angle is recorded and the three cameras at the front of the vehicle 
capture images of the scene.

| left camera | center camera | right camera |
| ----------- | ------------- | ------------ |  	
| ![][left]   | ![][center]   | ![][right]   |         

Not only that the side cameras can be used to extend the data by a factor of 3, they also 
provide important information of a possible car shift from the center of the lane. By 
treating the scene captured by a side camera as if it were viewed from the center of the 
vehicle, an artificial shift can be created. If the measured steering is then adjusted in a 
way that it would steer the car back to the lane center, data can be generated that may help 
the car to recover from poor situations. 

Let's take a look at the histograms of the collected data for both tracks. On the left side, 
only the images of the middle camera are used. On the right side, the dataset is the same. 
However, for each scene captured, the images from all three cameras are drawn randomly with 
the same probability. The number of samples remains constant. When an image from a side camera 
is drawn, the steering label is adjusted by a value of 0.2 (left camera +0.2, right camera -0.2)

| using only the center camera | using all cameras with equal probability |  	
| ------------------ | --------------- |  	
| ![][lake_center]   | ![][lake_all]   |
| ![][jungle_center] | ![][jungle_all] |

The following statements can be made:

- The lake track has a slight left shift. However, by having added a counter-clockwise lap 
  this is already mitigated somewhat.
- If only the middle camera is used, most of the steering values for the lake track are small. 
  That is, there are large portions of straight sections on the track. This can cause problems 
  in navigating the vehicle through turns, since the network is mainly taught to drive straight.
- By randomly sampling from all three cameras (and adjusting the steering for the side cameras),
  this bias can be reduced by stretching the distribution.
- The jungle track is longer and therefore provides more data.
- It also has many more turns and requires aggressive steering. It can already be seen that it 
  would be difficult to master the jungle track if the network were trained using only the data 
  from the lake track, since the lake track does not contain the sharp turns to learn from.
 
Using the three cameras is one way to extend the dataset to help the network generalize. We now 
want to introduce some more augmentations to further extend our generalization tools.

## Data Augmentation
The following augmentation methods have been implemented in [``augmentation.py``][augmentation] 
and may be used to help the network generalize.

```python
def choose_camera(sample, camera='random', probs=None):
    """
    Choose an image for a specific camera and eventually adjust the steering.

    If the left or right cameras are chosen the steering is adjusted.

    Args:
        sample: 
            a sample (row) of the data dataframe. Usually drawn of a batch by the generator
        camera: 
            'random', 'left', 'center' or 'right'. 
        probs: 
            the probabilities to choose the left, center or right cameras. 
            If None, the probabilities are uniform.
    Returns:
        (image, steering) tuple
    """

    if camera == 'random':
        camera = np.random.choice(["left", "center", "right"], p=probs)
    image = mpimg.imread(image_path(sample, camera=camera))
    steering = sample["steer"]
    if camera == "left":
        steering += CAMERA_STEERING_CORRECTION  # CAMERA_STEERING_CORRECTION = 0.2
    elif camera == "right":
        steering -= CAMERA_STEERING_CORRECTION 
    return image, steering
```

```python
def flip(image, steering, prob=0.5):
    """ Flip the image and steering with the given probability.

    Args:
        image: 
            the image to flip
        steering: 
            the steering corresponding to the image
        prob: 
            the flip probability
    Returns:
        the augmented image
    """
    if np.random.random() < prob:
        image = cv2.flip(image, 1)
        steering *= -1
    return image, steering
```

```python
def shadow(rgb_image, prob=0.5):
    """ Add a shadow to the rgb image with the given probability.

    The shadow is created by converting the RGB image into HSV space and
    modifying the value channel in a random range. The area in which the value
    is modified is defined by a convex hull created for 6 randomly chosen points
    in the lower half of the image.

    Args:
        rgb_image: 
            the image to add the shadow to. Has to be in RGB space.
        prob: 
            the probability to add the shadow
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
```
 
```python
def brightness(rgb_image, low=0.6, high=1.4, prob=0.5):
    """ Modify the brigtness of the rgb image with the given probability.

    The brightness is modified by converting the RGB image into HSV space and
    adusting the value channel in a random range between the low and high
    bounds.

    Args:
        rgb_image: 
            the image to modify the brightness. Has to be in RGB space.
        low: 
            lower value bound
        high: 
            upper value bound
        prob: 
            the probability to modify the brightness
    Returns:
        the augmented image
    """

    if np.random.random() < prob:
        hsv = rgb2hsv(rgb_image)
        value = hsv[:, :, 2]
        hsv[:, :, 2] = np.clip(value * np.random.uniform(low, high), 0, 255)
        rgb_image = hsv2rgb(hsv)
    return rgb_image
```

```python
def shift(image, steering, shiftx=60, shifty=20, prob=0.5):
    """ Shift the image and adjust the steering with the given probability.

    The steering of the shifted image is adjusted depending on the amount of
    pixels shifted in the width direction.

    Args:
        image: 
            the image to shift.
        steering: 
            the corresponding steering.
        shiftx: 
            the upper bound of pixels to shift in the width direction
        shifty: 
            the upper bound of pixels to shift in the height direction
        prob: 
            the probability to shift the image
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
```

In addition, the following preprocessing functions are defined.

```python
def crop_image(image, top=60, bot=25):
    """ Crop the upper and lower borders of the given image.

    Args:
        image: 
            the image to crop
        top: 
            the pixels to crop from the upper part
        bot: 
            the pixels to crop from the bottom part
    Returns:
        the cropped image
    """
    return image[top:-bot, :, :]
```

```python
def resize_image(image, shape=NVIDIA_SHAPE[0:2]):
    """ Resize the image to shape.

    Args:
        image: 
            input image
        shape: 
            (height, width) tuple, defaults to Nvidia input shape (66, 200)
    Returns:
        the resized image
    """
    h, w = shape
    return cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_AREA)
```

```python
def rgb2yuv(rgb_image):
    """ Convert the RGB image to YUV space. """
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
```

The cropping function is useful to let the network focus on the relevant part of the image, 
the road. Resizing and color space conversion are used to transform the image into the 
format used in the Nvidia paper.

Let's take a look on the augmentations and their influence on the steering for some randomly 
selected images.

![][augcrop]

## Training and Validation

- The collected data is split into 80% training and 20% validation data. 
- An Adam optimizer with a default learning rate of 0.001 is used. 
- The loss criterion chosen for this regression task is the MSE loss. It measures how well 
  the network predicts the steering for each image sample provided. 
- The network is trained for 10 epochs with a batch size of 128 samples. 
- A python generator is defined to provide the training and validation batches on the 
  fly during training or validation. No need to load the complete data into the memory.
- Each sample of the available training data is used exactly once per epoch. However, it may 
  be randomly augmented.
- The training is performed with the desired augmentation probabilities and the desired 
  preprocessing steps.
- Validation, on the other hand, is performed using only the unaugmented center camera image. 
  Depending on the investigation pipeline, it may also be preprocessed.

Some of the investigations performed to teach the network to drive the car autonomously around 
both tracks are listed in the table below. The complete table can be found in the [``models``][models] 
folder.

![][investigations]
&nbsp;


**Lessons learned**:

- **V0**: The network is able to drive around the lake track using only the center camera and 
  the raw collected training data on that track.
- **V1**: Adding augmentation further improves the driving capabilities.
- **V10**: However, on the jungle track, using only the center camera and augmentation is not 
  sufficient.
- **V11**: It is necessary to preprocess the data. This allows the network to focus on the road, 
  and reduces the distraction by the environment.
- **V13**: Training the network with the parameters that were successful on the jungle track 
  but now using the data of both tracks, suffices to master the lake track. However, since 
  the lake track is less curvy, the network "loses" some of its aggressive steering 
  capabilities that it learned on the jungle track.
- **V14**: Using all three cameras with the same probability increases the steering variation 
  and improves the network's driving. However, it is still not sufficient to master the sharp 
  turns of the jungle track.
- **V16**: By giving more weight to the lateral cameras, the network regains its ability to steer 
  agressivly when needed. It is now capable to drive around both tracks at maximum speed.

I guess it would beat me in a competition. :)


| ``videos/lake_V16.mp4`` | ``videos/jungle_V16.mp4`` | 	
| ---------- | -------------|  	
| ![][lake_v16] | ![][jungle_v16] |


## References
[[1]][NvidiaPaper2016] End to End Learning for Self-Driving Cars, 2016

[[2]][NvidiaPaper2017] Explaining How a Deep Neural Network Trained with End-to-End Learning 
Steers a Car, 2017


[cnn]: https://github.com/pabaq/CarND-Behavioral-Cloning/raw/master/images/NvidiaCNN.png "Nvidias CNN"
[simulator]: https://github.com/pabaq/CarND-Behavioral-Cloning/raw/master/images/Simulator.png "Simulator"
[left]: https://github.com/pabaq/CarND-Behavioral-Cloning/raw/master/images/left.jpg "left camera"
[center]: https://github.com/pabaq/CarND-Behavioral-Cloning/raw/master/images/center.jpg "center camera"
[right]: https://github.com/pabaq/CarND-Behavioral-Cloning/raw/master/images/right.jpg "right camera"
[lake_center]: https://github.com/pabaq/CarND-Behavioral-Cloning/raw/master/images/histogram_lake_center.png "Histogram Lake, center camera"
[lake_all]: https://github.com/pabaq/CarND-Behavioral-Cloning/raw/master/images/histogram_lake_all.png "Histogram Lake, all cameras"
[jungle_center]: https://github.com/pabaq/CarND-Behavioral-Cloning/raw/master/images/histogram_jungle_center.png "Histogram Jungle, center camera"
[jungle_all]: https://github.com/pabaq/CarND-Behavioral-Cloning/raw/master/images/histogram_jungle_all.png "Histogram Jungle, all cameras"
[augcrop]: https://github.com/pabaq/CarND-Behavioral-Cloning/raw/master/images/augmentation.png "Augmentation and Crop"
[investigations]: https://github.com/pabaq/CarND-Behavioral-Cloning/raw/master/images/investigations.png "Investigation pipelines"
[lake_v16]: https://github.com/pabaq/CarND-Behavioral-Cloning/raw/master/videos/lake_V16.gif "Autonomous Lake Complition"
[jungle_v16]: https://github.com/pabaq/CarND-Behavioral-Cloning/raw/master/videos/jungle_V16.gif "Autonomous Jungle Complition"

[model]: https://github.com/pabaq/CarND-Behavioral-Cloning/blob/master/model.py
[utilities]: https://github.com/pabaq/CarND-Behavioral-Cloning/blob/master/utilities.py
[augmentation]: https://github.com/pabaq/CarND-Behavioral-Cloning/blob/master/augmentation.py
[main]: https://github.com/pabaq/CarND-Behavioral-Cloning/blob/master/Behavioural_Cloning.py
[drive]: https://github.com/pabaq/CarND-Behavioral-Cloning/blob/master/drive.py
[models]: https://github.com/pabaq/CarND-Behavioral-Cloning/tree/master/models
[prep]: https://github.com/pabaq/CarND-Behavioral-Cloning/blob/master/Behavioural_Cloning.py#L25

[Course]: https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
[Project]: https://github.com/udacity/CarND-Behavioral-Cloning-P3
[UdacitySimulator]: https://github.com/udacity/self-driving-car-sim#term-1

[Nvidia2016]: https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
[NvidiaPaper2016]: https://arxiv.org/abs/1604.07316
[NvidiaPaper2017]: https://arxiv.org/abs/1704.07911
