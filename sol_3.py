import numpy as np


def _reduceImage(image):



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
    pass


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


if __name__ == '__main__':
    pass