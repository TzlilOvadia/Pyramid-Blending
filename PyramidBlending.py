import os
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import pickle
RGB = 2
GRAY_SCALE = 1
RGB_FORMAT = 3
EVEN_PIXELS = 2

def _buildGaussianVec(sizeOfVector):
    """
    Helper function to generate the gaussian vector with the size of the sizeOfVector
    """
    if sizeOfVector <= 2:
        return np.array([np.ones(sizeOfVector)])
    unitVec = np.ones(2)
    resultVec = np.ones(2)
    for i in range(sizeOfVector - 2):
        resultVec = scipy.signal.convolve(resultVec, unitVec)
    return np.array(resultVec/np.sum(resultVec)).reshape(1, sizeOfVector)



def _reduceImage(image, filter_vec):
    """
    a simple function to reduce the image size after blurring it
    :param image: image to reduce
    :return: reduced image
    """
    # Step 1: Blur the image:
    blurredImage = _blurImage(filter_vec, image)

    # Step 2: Sub-sample every 2nd pixel of the image, every 2nd row, from the blurred image:
    reducedImage = blurredImage[::EVEN_PIXELS,::EVEN_PIXELS]
    return reducedImage


def _blurImage(filter_vec, image):
    #Step 1: blur the rows:
    blurredImage = scipy.ndimage.filters.convolve(image,filter_vec)

    # Step 2: complete the blurred image:
    blurredImage = scipy.ndimage.filters.convolve(blurredImage, filter_vec.T)

    return blurredImage


def _expandImage(image, filter_vec):
    """

    :param image:  image to expand
    :param filter_vec:
    :return:
    """
    # Step 1: Expand the image using zeros on odd pixels:
    expandedImage = np.zeros((image.shape[0]*2,image.shape[1]*2))
    expandedImage[::EVEN_PIXELS,::EVEN_PIXELS] = image

    # Step 2: Blur the expanded image:
    blurredExpandedImage = _blurImage(filter_vec*2,expandedImage)
    return blurredExpandedImage


def _max_levels_calc(max_levels, im):
    """
    Simple helper function to calculate the maximum levels of the pyramid, given the
    restrictions on the assignment's pdf
    :return: the correct maximum levels of the pyramid we will calculate
    """
    widthLayers = np.log2(im.shape[1]/16)
    heightLayers = np.log2(im.shape[0]/16)
    max_levels = min(max_levels, int(widthLayers) + 1,
        int(heightLayers) + 1)
    return max_levels

def build_gaussian_pyramid(im, max_levels, filter_size):
    """

    :param im: a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the
    representation set to 1).

    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
                        to be used in constructing the pyramid filter
                        (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]). You may assume the filter size will be >=2.
    :return:
    """

    pyr = []
    gaussian_vec = _buildGaussianVec(filter_size)
    max_levels = _max_levels_calc(max_levels, im)
    tmp = im

    for i in range(max_levels):
        pyr.append(tmp)
        tmp = _reduceImage(np.copy(tmp), gaussian_vec)

    return pyr, gaussian_vec



def build_laplacian_pyramid(im, max_levels, filter_size):
    """

    :param im: a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the
    representation set to 1).
    :param max_levels: the maximal number of levels1 in the resulting pyramid.
    :param filter_size:
    :return:
    """
    #Step 1: calculate the gaussian pyramid so we can use it for next calculations:
    gauPyramid,filter_vec = build_gaussian_pyramid(im, max_levels, filter_size )

    #Step 2: Initialize the pyramid and add to each of its levels the correct value (using the formula we learnd):
    pyr = []
    for i in range(len(gauPyramid)-1):
        pyr.append(gauPyramid[i] - _expandImage(gauPyramid[i+1], filter_vec))

    #Step 3: add the last level of the pyramid:
    pyr.append(gauPyramid[-1])

    return pyr, filter_vec



def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    :param lpyr: laplacian pyramid
    :param filter_vec: filter vector
    :param coeff: python array of coefficients
    :return: reconstructed image from the lpyr
    """
    #Step 1: initialize the image we want to return:
    resultImage = lpyr[-1]*coeff[-1]

    #Step 2: iterate through all the levels of the pyramid and reconstruct an image out of it:
    for i in range(2, len(lpyr) + 1):
        resultImage = coeff[-i]*lpyr[-i] + _expandImage(resultImage,filter_vec)

    return resultImage


def _strech_image(im):
    """
    Simple function used to stretch the image to fit values in range [0,1]
    :param im:
    :return:
    """
    return np.floor(255*(im-np.min(im)/(np.max(im)-np.min(im))))

def render_pyramid(pyr, levels):
    """

    :param pyr:
    :param levels:
    :return:
    """
    #Step 1: initialize the values for the rendered pyramid dimensions:
    width = 0
    height = pyr[0].shape[0]
    levels = min(levels, len(pyr))
    for layer in range(levels):
        width += pyr[layer].shape[1]

    #Step 2: Create a blank image with the correct dimensions from step 1:
    res = np.zeros((height,width))
    currentWidthIdx = 0

    #Step 3: Fill the result image with each level from the pyramid:
    for i in range(levels):
        res[:pyr[i].shape[0],currentWidthIdx:currentWidthIdx+pyr[i].shape[1]] = _strech_image(pyr[i])
        currentWidthIdx += pyr[i].shape[1]

    return res


def display_pyramid(pyr, levels):
    """
    :param pyr:
    :param levels:
    :return:
    """
    res = render_pyramid(pyr,levels)
    plt.imshow(res, cmap= 'gray')
    plt.show()


def read_image(filename, representation):
    """
    filename - the filename of an image on disk (could be grayscale or RGB).
    representation - representation code, either 1 or 2 defining whether the output should be a:
    grayscale image (1)
    or an RGB image (2).
    NOTE: If the input image is grayscale, we won’t call it with represen- tation = 2.
    :param filename: String - the address of the image we want to read
    :param representation: Int - as described above
    :return: an image in the correct representation
    """
    if representation != RGB and representation != GRAY_SCALE:
        return "Invalid Input. You may use representation <- {1, 2}"
    tempImage = plt.imread(filename)[:, :, :3]
    resultImage = np.array(tempImage)

    if representation == GRAY_SCALE:
        resultImage = rgb2gray(tempImage)

    elif representation == RGB:
        resultImage = tempImage
    if resultImage.max() > 1:
        resultImage = resultImage/256

    return resultImage.astype(np.float64)


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """

    :param im1,im2: are two input grayscale images to be blended.
    :param mask:is a boolean (i.e. dtype == np.bool) mask containing
                True and False representing which parts of im1 and im2 should appear in the resulting im_blend.
                Note that a value of True corresponds to 1, and False corresponds to 0.
    :param max_levels: is the max_levels parameter you should use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im:  is the size of the Gaussian filter (an odd scalar that represents a squared filter) which
                            defining the filter used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: is the size of the Gaussian filter(an odd scalar that represents a squared filter)
                             which defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: Blended image
    """
    resImg = np.zeros(im1.shape)
    mask = mask.astype(np.float64)
    # Build Gaussian pyramid for the mask:
    G, filter_vec = build_gaussian_pyramid(mask, max_levels, filter_size_mask)
    if len(im1.shape) == RGB_FORMAT:
        for channel in range(len(im1.shape)):
            # Build laplacian pyramid for im1 and im2:
            L_a = build_laplacian_pyramid(im1[:,:,channel],max_levels, filter_size_im)[0]
            L_b = build_laplacian_pyramid(im2[:,:,channel], max_levels, filter_size_im)[0]


            # Build laplacian pyramid using the formula we saw in class:
            L_c = np.multiply(G,L_a) + np.multiply(np.subtract(1, G),L_b)
            resImg[:,:,channel] = laplacian_to_image(L_c,filter_vec,np.ones(max_levels))
    else:
        # Build laplacian pyramid for im1 and im2:
        L_a = build_laplacian_pyramid(im1, max_levels, filter_size_im)[0]
        L_b = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]

        # Build laplacian pyramid using the formula we saw in class:
        L_c = np.multiply(G, L_a) + np.multiply(np.subtract(1, G), L_b)
        resImg = laplacian_to_image(L_c, filter_vec, np.ones(max_levels))

    return np.clip(resImg,0,1)


def blending_example1():
    """ Title:
    Game of CS
    """
    fig, ax = plt.subplots(2,2)
    im1 = read_image( relpath("external/lect.jpg"),RGB)
    im2 = read_image( relpath("external/final.jpg"),RGB)
    mask = read_image(relpath("external/masks-3.jpg"), GRAY_SCALE).astype(bool)
    blend_image = pyramid_blending(im1, im2, mask, 3, 30, 5)
    plt.imshow(blend_image)
    plt.show()

    ax[0,0].imshow(im1)
    ax[0,1].imshow(im2)
    ax[1,0].imshow(mask,cmap='gray')
    ax[1,1].imshow(blend_image)
    ax[1,1].set_title("Games of CS (Mis'hakey Ha CS)")
    plt.show()

    return im1, im2, mask, blend_image


def blending_example2():

    fig,ax = plt.subplots(2,2)
    im1 = read_image(relpath("external/koch.jpg"),RGB)
    im2 = read_image(relpath("external/markoch.jpg"),RGB)
    mask = read_image(relpath("external/maskoo.jpg"), GRAY_SCALE).astype(bool)
    blend_image = pyramid_blending(im1, im2, mask, 3, 30, 5)
    plt.imshow(blend_image)
    plt.show()
    ax[0,0].imshow(im1)
    ax[0,1].imshow(im2)
    ax[1,0].imshow(mask,cmap='gray')
    ax[1,1].imshow(blend_image)
    ax[1, 1].set_title("Markoch")
    plt.show()

    return im1, im2, mask, blend_image


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

if __name__ == "__main__":
    im1_a, im2_a, mask_a, blend_image_a = blending_example1()
    im1_b, im2_b, mask_b, blend_image_b = blending_example2()
