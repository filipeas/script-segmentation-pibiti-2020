import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage import io,  img_as_float, img_as_ubyte, img_as_int
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
from skimage import measure
from skimage.morphology import convex_hull_image, convex_hull_object
from skimage.morphology import square, dilation
import cv2 as cv



img = io.imread('lesao.png')

img_gray = rgb2gray(img)

hull = convex_hull_object(img)

#hull_dilated = dilation(hull, square(28 ))
contours = measure.find_contours(hull, 0.5)
#order the contours in decrescent order and get the biggest one
biggest_contour = sorted(contours, key=lambda x: len(x), reverse=True)[0]



#s = np.linspace(0, 2*np.pi, 400)
#r = 100 + 100*np.sin(s)
#c = 220 + 100*np.cos(s)
#init = np.array([r, c]).T

smoothed_img = gaussian(img_gray, 4)
snake = active_contour(smoothed_img,
                       biggest_contour, alpha=0.001, beta=20, gamma=0.001, w_edge=0)

gimage = inverse_gaussian_gradient(img)
evolution = []
#callback = store_evolution_in(evolution)
ls = morphological_chan_vese(gimage, 230,  'circle',
                                       smoothing=3, lambda1=1, lambda2=1)       

for i in ls:
 #   print(i)
    img[i[0]][i[1]] = 255                                                     

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(snake[:, 1], snake[:, 0], '-r', lw=2)
ax.contour(ls, [0.5], colors='b')
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

plt.show()