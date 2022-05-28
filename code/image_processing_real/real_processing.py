import numpy as np

import cv2
import numpy as np
from skimage.morphology import reconstruction
from PIL import Image

def imfill(img):
    # https://stackoverflow.com/questions/36294025/python-equivalent-to-matlab-funciton-imfill-for-grayscale
    # Use the matlab reference Soille, P., Morphological Image Analysis: Principles and Applications, Springer-Verlag, 1999, pp. 208-209.
    #  6.3.7  Fillhole
    # The holes of a binary image correspond to the set of its regional minima which
    # are  not  connected  to  the image  border.  This  definition  holds  for  grey scale
    # images.  Hence,  filling  the holes of a  grey scale image comes down  to remove
    # all  minima  which  are  not  connected  to  the  image  border, or,  equivalently,
    # impose  the  set  of minima  which  are  connected  to  the  image  border.  The
    # marker image 1m  used  in  the morphological reconstruction by erosion is set
    # to the maximum image value except along its border where the values of the
    # original image are kept:

    seed = np.ones_like(img)*255
    img[ : ,0] = 0
    img[ : ,-1] = 0
    img[ 0 ,:] = 0
    img[ -1 ,:] = 0
    seed[ : ,0] = 0
    seed[ : ,-1] = 0
    seed[ 0 ,:] = 0
    seed[ -1 ,:] = 0

    fill_img = reconstruction(seed, img, method='erosion')
    return fill_img


def change_size(image, size):
    return 





if __name__ == '__main__':
    image = np.load('image.npy')
    depth_image = image[:,:,3]
    depth_image_show = Image.fromarray(depth_image)
    #filled_image.save('d.png')
    depth_image_show.save("raw.png")

    
    mask = np.where(depth_image < 254, 0, depth_image)
    
    dst = cv2.inpaint(depth_image, mask, 1, cv2.INPAINT_TELEA)  # cv::INPAINT_NS or cv::INPAINT_TELEA
    mask = Image.fromarray(mask)
    mask.save('mask.png')
    #filled_image = imfill(depth_image)
    filled_image = Image.fromarray(dst)
    filled_image.save('filtered.png')


