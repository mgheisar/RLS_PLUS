import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from pathlib import Path

# factor, iter = 8, 79
factor, iter = 16, 69
# factor, iter = 32, 29
# factor, iter = 64, 19
indices = [128, 194, 219, 272, 289, 311, 319]
index = 319
# for index in range(600,1000):
# if not Path.exists(Path(f"input/project/resHR/{index:05d}_64x_HR.jpg")):
#     continue
image_list = [f"input/project/resLR_{factor}x/{index:05d}_{factor}x.jpg",
              f"input/project/resSR/{factor}x_base_pulse/{index:05d}_{factor}x.jpg",
              f"input/project/resSR/{factor}x_base_brgm/{index:05d}_{factor}x.jpg",
              f"/projects/superres/Marzieh/ddrm/exp/image_samples/{factor}x/{index:05d}_8x.jpg",
                f"input/project/resSR/GFPGAN_{factor}/{index:05d}.jpg",
                f"input/project/resSR/GPEN_{factor}/{index:05d}.jpg",
              f"input/project/resSR/{factor}x_base/{index:05d}_{factor}x.jpg",
              f"input/project/resSR/RLSPlus/train/{factor}x/{index:05d}_{factor}x_boost{iter}.jpg",
              f"input/project/resHR/{index:05d}_64x_HR.jpg"]
for ii, image in enumerate(image_list):
    if ii == 1 or ii == 2 or ii == 3 or ii == 6:
        continue
    img = cv2.imread(image)
    if ii == 0:
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    # x1, y1 = 400, 580  # mouth 1 00089, 00045,00057,
    # x1, y1 = 250, 310  # left eye 00675, 00076
    # x1, y1 = 280, 340  # left eye 00036
    # x1, y1 = 450, 450  # nose
    # x1, y1 = 560, 340  # right eye 01191, 02458
    # x1, y1 = 640, 290  # right eye
    # x1, y1 = 460, 590  # mouth 00022,
    # x1, y1 = 380, 670  # 453
    # left eye to right eye
    # x1, y1 = 280, 390  ## w, h = 470, 150 # eyes 128
    # x1, y1 = 290, 720  # w, h = 470, 150 mouth 194
    # x1, y1 = 590, 490 # w, h = 270, 300 # cheeks 219
    # x1, y1 = 280, 40 # w, h = 470, 150 # forehead 289
    # x1, y1 = 2, 520  # w, h = 270, 300 # cheeks 311
    x1, y1 = 410, 600  # w, h = 270, 300 # mouth 319
    w, h = 270, 300
    # w, h = 200, 200
    roi_region = ((y1, y1 + h), (x1, x1 + w))
    roi_im = img[roi_region[0][0]:roi_region[0][1], roi_region[1][0]:roi_region[1][1]].copy()

    # draw rectangle
    cv2.rectangle(img, (roi_region[1][1], roi_region[0][0]), (roi_region[1][0], roi_region[0][1]), (0, 0, 255), 2)
    #
    # zoom in roi
    SCALE = 2
    roi_im = cv2.resize(roi_im, (roi_im.shape[1] * SCALE, roi_im.shape[0] * SCALE))
    img[0:roi_im.shape[0], 0:roi_im.shape[1]] = roi_im
    cv2.rectangle(img, (0, 0), (roi_im.shape[1], roi_im.shape[0]), (0, 0, 255), 2)

    # plt.imshow(img[:, :, ::-1])
    # plt.show()
    # # # cv2.imshow("image", img)
    # # # cv2.waitKey(0)
    im_name = Path(image).stem
    method = ""
    if ii == 0:
        method = "_LR"
    elif ii == 1:
        method = f"_pulse"
    elif ii == 2:
        method = f"_brgm"
    elif ii == 3:
        method = "_ddrm"
    elif ii == 4:
        method = "_GFPGAN"
    elif ii == 5:
        method = "_GPEN"
    elif ii == 6:
        method = "_RLS"
    elif ii == 7:
        method = "_RLS+"
    elif ii == 8:
        method = "HR"
    print(im_name)
    cv2.imwrite(f"input/project/resSR/res_iccv_{factor}x/{im_name}{method}.jpg", img)
    # cv2.imwrite(f"input/project/resSR/res_p/{im_name}{method}.jpg", img)
