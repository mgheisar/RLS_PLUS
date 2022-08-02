# import numpy as np
# from skimage import io
# import scipy.ndimage as ndimage
# from skimage.feature import peak_local_max
# import scipy.ndimage.filters as filters
# import matplotlib.pyplot as plt
#
# fname = "/projects/superres/Marzieh/stylegan2/input/HR/img0.png"
# neighborhood_size = 3
# threshold = 217
#
# data = io.imread(fname)
# data_max = filters.maximum_filter(data, neighborhood_size)
# maxima = (data == data_max)
# data_min = filters.minimum_filter(data, neighborhood_size)
# diff = ((data_max - data_min) > threshold)
# maxima[diff == 0] = 0
#
# labeled, num_objects = ndimage.label(maxima)
# xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
#
# xy = peak_local_max(data, min_distance=0, threshold_abs=254)
# plt.imshow(data)
# plt.autoscale(False)
# plt.plot(xy[:, 1], xy[:, 0], 'ro')
# plt.show()

from PIL import Image
import numpy as np
from findmaxima2d import find_maxima, find_local_maxima
import matplotlib.pyplot as plt

img_data = Image.open("/projects/superres/Marzieh/stylegan2/input/HR/img0.png")
img_data1 = np.array(img_data)
ntol = 30  # Noise Tolerance.

img_data = (np.sum(img_data1, 2) / 3.0)

# Finds the local maxima using maximum filter.
local_max = find_local_maxima(img_data)

# Finds the maxima.
y, x, regs = find_maxima(img_data, local_max, ntol)
plt.imshow(img_data1)
plt.plot(x, y, 'rx')
plt.show()