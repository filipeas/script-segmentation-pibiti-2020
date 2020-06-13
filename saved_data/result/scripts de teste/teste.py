import cv2
import numpy as np

# import numpy as np
from random import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# from SFc_means import * # classe responsável por implementar o algoritmo proposto, SFc-means
# from utils import * # classe responsável por guardar funções gerais
from scipy.ndimage.morphology import binary_fill_holes
from skimage.io import imread, imsave, imshow, show
from skimage import img_as_ubyte, data
from skimage.measure import label, find_contours
# from dilation_and_erosion import *
from skimage.morphology import disk, binary_closing, binary_erosion, binary_dilation, binary_opening, convex_hull_image, square
from skimage import data, img_as_float
from skimage.util import invert
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import active_contour

img=cv2.imread("tcc_result (imagem resultante da classificação).png")
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my masks
mask = mask0+mask1

# set my output img to zero everywhere except my mask
output_img = img.copy()
output_img[np.where(mask==0)] = 0

# or your HSV image, which I *believe* is what you want
output_hsv = img_hsv.copy()
output_hsv[np.where(mask==0)] = 0

# selecionando o maior elemento
shape = output_img.shape
labels = label(output_img) #get the image componentes labels 
unique, counts = np.unique(labels, return_counts=True) #count the number of pixels in each label
labeled_qtd = dict(zip(unique, counts)) #zip the labels and the quantities and put all this in a dict
del labeled_qtd[0] #remove the key corresponding to the background

sorted_labels = sorted(labeled_qtd, key=labeled_qtd.get, reverse=True)  #sort the labels in decrescent order to get the biggest label

#mount, based on the biggest label, an image with only the biggest component
biggest_component_image = np.zeros((shape[0], shape[1]), dtype=int)
for i in range(shape[0]):
    for j in range(shape[1]):
        if (labels[i][j] == sorted_labels[0]).any():
            biggest_component_image[i][j] = 255

output_img = biggest_component_image

imsave('output_img.png', img_as_ubyte(output_img))