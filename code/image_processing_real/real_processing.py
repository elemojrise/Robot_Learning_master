import numpy as np

import cv2
import numpy as np
from pandas import cut
from skimage.morphology import reconstruction
from PIL import Image

def imfill(depth_image):
    mask = np.where(depth_image < 254, 0, depth_image)
    dst = cv2.inpaint(depth_image, mask, 1, cv2.INPAINT_TELEA)  # cv::INPAINT_NS or cv::INPAINT_TELEA
    return dst

def change_size(image, size):
    resized_image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC)
    return resized_image

def cut_d(image):
    cut_image = image[110:1090,110:1090]
    return cut_image





if __name__ == '__main__':


    image = np.load('image.npy')
    depth_image = image[:,:,3]
    rgb_image = image[:,:,:3]


    cut_d = cut_d(depth_image)
    print("image size", cut_d.shape)

    cut_d = change_size(cut_d, [100,100])

    cut_d = imfill(cut_d)

    
    Image.fromarray(cut_d).save("images/cut_d_2.png")

    # Image.fromarray(depth_image).save("images/raw_d.png")
    # Image.fromarray(rgb_image).save("images/raw_rgb.png")


    # sized_d = change_size(depth_image,(83,83))
    # Image.fromarray(sized_d).save("images/sized_d.png")
    # Image.fromarray(change_size(rgb_image,(83,83))).save("images/sized_rgb.png")
    

    # Image.fromarray(imfill(sized_d)).save('images/sized_then_filled_d.png')

    # Image.fromarray(change_size(imfill(depth_image),(83,83))).save('images/filled_then_sized_d.png')




