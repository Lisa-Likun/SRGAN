import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
import skimage.io
import cv2

def get_images(filename, is_crop, fine_size, images_norm):
    # print ("filename=",filename)
    img = cv2.imread(filename)
    # print("?????????????shape of img=", (np.array(img)).shape)
    # img = img.reshape(480, 640,3)


    if is_crop:
        size = img.shape
        start_h = int((size[0] - fine_size)/2)
        start_w = int((size[1] - fine_size)/2)
        img = img[start_h:start_h+fine_size, start_w:start_w+fine_size,:]
    img = np.array(img).astype(np.float32)

    # print("!!!!!!!!!!!!!!!!shape of img_resize=", img_resize.shape)
    # print("!!!!!!!!!!!!!!!!shape of img=", img.shape)
    if images_norm:
        img = (img-127.5)/127.5
    return img

def save_images(images, filename):
    for i in range (len(images)):
        # image_path = filename
        plt.imshow(images[i], interpolation='nearest')
        plt.axis("off")
        plt.savefig(filename)


    # return scipy.misc.imsave(filename, merge_images(images, size))

def merge_images(images, size):
    h,w = images.shape[1], images.shape[2]
    imgs = np.zeros((size[0]*h,size[1]*w, 3))
    
    for index, image in enumerate(images):
        i = index//size[1]
        j = index%size[0]
        imgs[i*h:i*h+h, j*w:j*w+w, :] = image

    return imgs
