# Split a table of images 5*3 into 5*3 images and store the patches
# in a folder named "patches"
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob

image = cv2.imread("/projects/superres/Marzieh/SR_Stylegan/input/project/multi.jpg")
for i in range(5):
    for j in range(3):
        # patch size is 343*343
        # remove the margin of 10 pixels
        patch = image[i*343+10:(i+1)*343-10, j*343+10:(j+1)*343-10]
        # save the patch in a folder named "patches"
        cv2.imwrite(f"/projects/superres/Marzieh/SR_Stylegan/input/project/patches/patch{i}{j}.jpg", patch)