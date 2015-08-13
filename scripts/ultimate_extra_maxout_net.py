from __future__ import absolute_import 
from __future__ import print_function 
import os 
import numpy as np
import pandas as pd 
from skimage import io as io 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers.core import Dense, MaxoutDense, Dropout, Activation, Flatten 
from keras.layers.convolutional import Convolution2D, MaxPooling2D 
from keras.utils import np_utils, generic_utils 
from six.moves import range 

def create_model():
    model = Sequential()
    # First Convolutional Layers
    model.add(Convolution2D(32, 1, 3, 3, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))
    # Second Convolutional Layers
    model.add(Convolution2D(64, 32, 3, 3, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))
    # Third Convolutional Layers
    model.add(Convolution2D(128, 64, 3, 3, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))
    # Dense Fully Connected Layer with MaxOut
    model.add(Flatten())
    model.add(MaxoutDense(128 * 16 * 16, 512, init='he_normal', nb_feature=2))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Extra Dense Fully Connected Layer with MaxOut
    model.add(MaxoutDense(512, 512, init='he_normal', nb_feature=2))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Dense Fully Connected Layer
    model.add(Dense(512, nb_classes, init='he_normal'))
    model.add(Activation('softmax'))
    # Compiling
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # Returning
    return model

def sample(labels, size):
    return labels.ix[np.random.choice(labels.index, size, replace = False)]
    
def downsample(labels, levels, sizes):
    return pd.concat([sample(labels[labels.level == level], size) for level, size in zip(levels, sizes)]) 

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n] 

def load_train_data(levels, sizes, train_folder = '../train-128', image_size = 128, labels_file = '../data/trainLabels.csv'):
    labels = pd.read_csv(labels_file)
    list_of_files = downsample(labels, levels, sizes)
    list_of_files = list_of_files.reindex(np.random.permutation(list_of_files.index))
    train_file_list = [file + '.jpeg' for file in list_of_files.image.values];
    train_size = len(train_file_list)
    shape = (train_size, 1, image_size, image_size)
    x_train = np.zeros(shape, dtype = 'float32')
    y_train = np.zeros(train_size, dtype = 'uint8')
    for index, fn in enumerate(train_file_list):
        original_image = io.imread(train_folder + '/' + fn)
        x_train[index] = np.asarray(original_image.reshape(1, image_size, image_size), dtype = 'float32')
        y_train[index] = labels.ix[labels['image'] == fn.replace('.jpeg', ''), 'level'].values[0]
    return x_train, y_train 

def batch_load_test_data(test_file_list, test_folder = '../test-128', image_size = 128):
    test_size = len(test_file_list)
    shape = (test_size, 1, image_size, image_size)
    x_test = np.zeros(shape, dtype = 'float32')
    test_labels = np.empty(test_size, dtype='S12')
    for index, fn in enumerate(test_file_list):
        original_image = io.imread(test_folder + '/' + fn)
        original_image = (original_image - original_image.mean()) / original_image.std()
        x_test[index] = np.asarray(original_image.reshape(1, image_size, image_size), dtype = 'float32')
        test_labels[index] = fn.replace('.jpeg', '')
    return x_test, test_labels 

def test(model, test_folder = '../test-128', image_size = 128):
    test_file_list = [file for file in os.listdir(test_folder) if file.endswith('.jpeg')];
    output = pd.DataFrame(columns = ['image', 'level'])
    for test_files in chunks(test_file_list, 1000):
        x_test, test_labels = batch_load_test_data(test_files, test_folder, image_size)
        y_test = model.predict_classes(x_test)
        output = pd.concat([output, pd.DataFrame({ 'image': test_labels, 'level': y_test })])
        del x_test, y_test, test_labels
    return output 

def save_data(output, output_file = 'submission.csv'):
    output.sort(['image'], inplace = True)
    output['level'] = output['level'].astype(np.int8)
    output.to_csv(output_file, index = False)
    
if __name__ == "__main__":
    submission_file = 'submissionUltimateExtraMaxOut.csv' 
    weight_file = 'ultimateExtraMaxoutNet.hdf5'
    levels = [0, 1, 2, 3, 4]
    sizes = [700, 700, 700, 700, 700]

    np.random.seed(1337) # for reproducibility 
    
    batch_size = 32 
    nb_classes = 2 
    nb_epoch = 200
    model = create_model()
    X_train, Y_train = load_train_data(levels, sizes)
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    print(X_train.shape[0], 'train samples')
    
    print("Using real time data augmentation")
    
    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
                        featurewise_center=False,
                        samplewise_center=True,
                        featurewise_std_normalization=False,
                        samplewise_std_normalization=True,
                        zca_whitening=False,
                        rotation_range=45,
                        width_shift_range=0.4,
                        height_shift_range=0.4,
                        horizontal_flip=True,
                        vertical_flip=True)
    datagen.fit(X_train)
    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print("Training...")
        # batch train with realtime data augmentation
        progbar = generic_utils.Progbar(X_train.shape[0])
        for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size = batch_size):
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("train loss", loss)])
    model.save_weights(weight_file)
        
    output = test(model)
    save_data(output, output_file = submission_file)
