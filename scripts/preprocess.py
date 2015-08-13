import os  
import numpy as np
from skimage import io as io
from skimage import color
from skimage import transform
from skimage.exposure import equalize_hist, rescale_intensity 
from skimage.filters import gaussian_filter

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
    
def preprocess_images(input_folder = '../train', output_folder = '../train-256', pixel_size = 256.):
    input_list = set([file for file in os.listdir(input_folder) if file.endswith('.jpeg')])
    output_list = set([file for file in os.listdir(output_folder) if file.endswith('.jpeg')])
    files_to_process = list(input_list - output_list)
    
    for fn in files_to_process:
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
        output_image = remove_padding(original_image, kernel)
        grayed_image = color.rgb2gray(padded_image)
        blurred_image = gaussian_filter(grayed_image,sigma=6, multichannel=False)
        difference_image = equalize_hist(grayed_image - blurred_image)

        scale = difference_image.shape[0] / pixel_size
        if scale > 1:
            output_image = transform.pyramid_reduce(difference_image, downscale=scale)
        elif scale < 1:
            output_image = transform.pyramid_expand(difference_image, upscale=1/scale)
        else:
            output_image = difference_image

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

def scale_images(input_folder = '../train-256', output_folder = '../train-128', pixel_size = 128.):
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

if __name__ == '__main__':
    preprocess_images(input_folder = '../train', output_folder = '../train-256')
    scale_images(input_folder = '../train-256', output_folder = '../train-128', pixel_size = 128.)
    preprocess_images(input_folder = '../test', output_folder = '../test-256')
    scale_images(input_folder = '../test-256', output_folder = '../test-128', pixel_size = 128.)