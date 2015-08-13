Diabetic Retinopathy Detection
==============================
This documents the model I used to reach 127 / 673 (Public Leaderboard) or 119 / 673 (Private Leaderboard) on the [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection) competition on Kaggle. 

Prerequisites
-------------
- Python 2.7 [Anaconda distribution](http://continuum.io/downloads). This will have NumPy, SciPy, Pandas, Scikit-Image, and Scikit-Learn included.
- [CUDA](https://developer.nvidia.com/cuda-downloads) with [cuDNN](https://developer.nvidia.com/cudnn).
- [Theano](http://deeplearning.net/software/theano/).
- [Keras](http://keras.io/).

The original images should have been unpacked and located in separate ```train``` and ```test``` folders. Output of the processed images are saved in ```train-128``` and ```test-128``` folders. Scripts are in a ```scripts``` folder, model output is saved in a ```model``` folder.

Final Rank
----------
The competition is scored using the [Quadratic Weighted Kappa](https://www.kaggle.com/c/diabetic-retinopathy-detection/details/evaluation) metric.  
Public Leaderboard: 127 / 673 (0.41644),  
Private Leaderboard: 119 / 673 (0.42564).  
Top 25% badge.

Approach
--------
Preprocessing of the images are absolutely necessary; my convolutional neural networks don't learn on the original color image. I first made sure all images are square instead of rectangular, so I cropped the sides (the additional sides were, as far as I can see, all black anyway). 
```python
from skimage import io as io

def padding_for_kernel(kernel):
    """ Return the amount of padding needed for each side of an image.

    For example, if the returned result is [1, 2], then this means an
    image should be padded with 1 extra row on top and bottom, and 2
    extra columns on the left and right.
    """
    # Slice to ignore RGB channels if they exist.
    image_shape = kernel.shape[:2]
    # We only handle kernels with odd dimensions so make sure that's true.
    # (The "center" pixel of an even number of pixels is arbitrary.)
    assert all((size % 2) == 1 for size in image_shape)
    return [(size - 1) // 2 for size in image_shape]

def add_padding(image, kernel):
    h_pad, w_pad = padding_for_kernel(kernel)
    return np.pad(image, ((h_pad, h_pad), (w_pad, w_pad)), mode='constant', constant_values=0)

def remove_padding(image, kernel):
    inner_region = []  # A 2D slice for grabbing the inner image region
    for pad in padding_for_kernel(kernel):
        slice_i = slice(None) if pad == 0 else slice(pad, -pad)
        inner_region.append(slice_i)
    return image[inner_region]
    
for fn in os.listdir(input_folder):
    original_image = io.imread(input_folder + '/' + fn)
    shape = original_image.shape[1] - original_image.shape[0]
    if shape < 0:
        shape = -shape
        if shape % 2 == 0:
            shape += 1
        kernel = np.ones((shape, 1))
    else:
        if shape % 2 == 0:
            shape += 1
        kernel = np.ones((1, shape))
    print fn, original_image.shape, kernel.shape
    output_image = remove_padding(original_image, kernel)
    io.imsave(output_folder + '/' + fn, output_image)
```
Then I transformed the images to grayscale and substracted a Gaussian filtered image to highlight the features.
```python
from skimage.exposure import equalize_hist, rescale_intensity 
from skimage.filters import gaussian_filter

grayed_image = color.rgb2gray(padded_image)
blurred_image = gaussian_filter(grayed_image,sigma=6, multichannel=False)
difference_image = equalize_hist(grayed_image - blurred_image)
```
Finally I scaled all images down/up, first to 256x256 and then an additional step to 128x128.
```python
from skimage import io as io
from skimage import transform

input_list = set([file for file in os.listdir(input_folder) if file.endswith('.jpeg')])
output_list = set([file for file in os.listdir(output_folder) if file.endswith('.jpeg')])
files_to_process = list(input_list - output_list)

for fn in files_to_process:
    original_image = io.imread(input_folder + '/' + fn)
    scale = original_image.shape[0] / pixel_size

    if scale > 1:
        output_image = transform.pyramid_reduce(original_image, downscale=scale)
    elif scale < 1:
        output_image = transform.pyramid_expand(original_image, upscale=1/scale)
    else:
        output_image = original_image
    print input_folder + '/' + fn, "\t-> ", output_image.shape

    if output_image.shape != (pixel_size, pixel_size):
        x = pixel_size - output_image.shape[0]
        y = pixel_size - output_image.shape[1]
        if x < 0:
            image = output_image[:x, :]
        if y < 0:
            image = output_image[:, :y]
    
    output_image = (output_image - output_image.min()) / output_image.max()
    
    try:
        io.imsave(output_folder + '/' + fn, output_image)
    except ValueError:
        print fn + ' not saved!'
```
As input for the convolutional neural networks, I downsample the data to 700 samples per class (class 4 is least present, with 708 images). After this, shuffle the input data.
```python
def sample(labels, size):
    return labels.ix[np.random.choice(labels.index, size, replace = False)]
    
def downsample(labels, levels, size):
    return pd.concat([sample(labels[labels.level == level], size) for level in levels])
```
The highest scoring model on this input data was (public QWK = 0.41081, private QWK = 0.41960):
```python
model.add(Convolution2D(32, 1, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 32, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(Convolution2D(64, 64, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 64, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(Convolution2D(128, 128, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(MaxoutDense(128 * 16 * 16, 512, init='he_normal', nb_feature=2))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512, nb_classes, init='he_normal'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```
trained for 200 epochs with a micro batch size of 32. This will take approximately 6 hours on a [g2.2xlarge](http://aws.amazon.com/de/ec2/instance-types/) on AWS EC2.

The image augmentation is 
```python
datagen = ImageDataGenerator(
    featurewise_center = False,            # set input mean to 0 over the dataset
    samplewise_center = True,              # set each sample mean to 0
    featurewise_std_normalization = False, # divide inputs by std of the dataset
    samplewise_std_normalization = True,   # divide each input by its std
    zca_whitening = False,                 # apply ZCA whitening
    rotation_range = 45,                   # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range = 0.4,               # randomly shift images horizontally (fraction of total width)
    height_shift_range = 0.4,              # randomly shift images vertically (fraction of total height)
    horizontal_flip = True,                # randomly flip images horizontally
    vertical_flip = True)                  # randomly flip images vertically
```
The other models in the ensemble were (public QWK = 0.38449, private QWK = 0.39447):
```python
model.add(Flatten())
model.add(MaxoutDense(128 * 16 * 16, 512, init='he_normal', nb_feature=2))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(MaxoutDense(512, 512, init='he_normal', nb_feature=2))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512, nb_classes, init='he_normal'))
model.add(Activation('softmax'))
```
and (public QWK = 0.36963, private QWK = 0.37533):
```python
datagen = ImageDataGenerator(
    featurewise_center = False,            # set input mean to 0 over the dataset
    samplewise_center = True,              # set each sample mean to 0
    featurewise_std_normalization = False, # divide inputs by std of the dataset
    samplewise_std_normalization = True,   # divide each input by its std
    zca_whitening = False,                 # apply ZCA whitening
    rotation_range = 180,                  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range = 0.4,               # randomly shift images horizontally (fraction of total width)
    height_shift_range = 0.4,              # randomly shift images vertically (fraction of total height)
    horizontal_flip = True,                # randomly flip images
    vertical_flip = True)                  # randomly flip images
```
Then finally we construct a majority vote ensemble (public QWK = 0.41644, private QWK = 0.42564):
```python
from collections import Counter

labels = []
y_test = []

one = pd.read_csv('../model/submissionUltimate.csv')
two = pd.read_csv('../model/submissionUltimate.csv')
three = pd.read_csv('../model/submissionUltimateExtraMaxOut.csv')
four = pd.read_csv('../model/submissionUltimateMoreAug.csv')

for i in range(len(one)):
    c = Counter([one.level[i], two.level[i], three.level[i], four.level[i]])
    y_test.append(c.most_common()[0][0])
    labels.append(one.image[i])
output = pd.DataFrame({ "image": labels, "level": y_test })
output.sort(['image'], inplace = True)
output['level'] = output['level'].astype(np.int8)
output.to_csv('../model/top_majority_ensemble.csv', index=False)
```