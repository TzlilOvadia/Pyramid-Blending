import numpy as np
import scipy.signal


def _buildGaussianVec(sizeOfVector):
    if sizeOfVector <= 2:
        return np.array([np.ones(sizeOfVector)])
    unitVec = np.ones(2)
    resultVec = np.ones(2)
    for i in range(sizeOfVector - 2):
        resultVec = scipy.signal.convolve(resultVec, unitVec)
    return resultVec/np.sum(resultVec)

def _reduceImage(image, filter_vec):
    """

    :param image: image to reduce
    :return: reduced image
    """
    # Step 1: Blur the image:
    blurredImage = scipy.ndimage.filters.convolve(filter_vec,image)
    blurredImage += scipy.ndimage.filters.convolve(filter_vec.T,blurredImage)

    # Step 2: Sub-sample every 2nd pixel of the image, every 2nd row, from the blurred image:
    reducedImage = blurredImage[::2,::2]
    return reducedImage

def _expandImage(image, filter_vec):
    """

    :param image:  image to expand
    :param filter_vec:
    :return:
    """
    



def build_gaussian_pyramid(im, max_levels, filter_size):
    """

    :param im: a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the
    representation set to 1).

    :param max_levels: the maximal number of levels1 in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
                        to be used in constructing the pyramid filter
                        (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]). You may assume the filter size will be >=2.
    :return:
    """

    pyr = [im]
    gaussian_vec = _buildGaussianVec(filter_size)

    for i in range(max_levels):
        im = _reduceImage(im, gaussian_vec)
        pyr.append(im)

    return pyr


def build_laplacian_pyramid(im, max_levels, filter_size):
    """

    :param im: a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the
    representation set to 1).
    :param max_levels: the maximal number of levels1 in the resulting pyramid.
    :param filter_size:
    :return:
    """
    pass



def laplacian_to_image(lpyr, filter_vec, coeff):
    pass



def render_pyramid(pyr, levels):
    pass


def display_pyramid(pyr, levels):
    pass


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    pass

