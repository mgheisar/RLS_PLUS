import torchvision
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from PIL import Image
import glob
from pathlib import Path

# # environment: ddrm

portrait_enhancement = pipeline(Tasks.image_portrait_enhancement, model='damo/cv_gpen_image-portrait-enhancement-hires')
factor = 16
img_list = sorted(glob.glob(f"input/project/resLR_{factor}x/*.jpg"))
for path_img in img_list:
    img = cv2.imread(path_img)
    # resize to 128x128
    img = cv2.resize(img, (512, 512))
    cv2.imwrite('input.jpg', img)
    result = portrait_enhancement(f"input.jpg")
    img_result = cv2.resize(result[OutputKeys.OUTPUT_IMG], (1024, 1024))
    img_name = (Path(path_img).stem.split('_')[0] + ".jpg")
    cv2.imwrite(f'input/project/resSR/GPEN_{factor}/{img_name}', img_result)
    # img_result = cv2.resize(result[OutputKeys.OUTPUT_IMG], (128, 128))
    # cv2.imwrite(f'input/project/resSR/GPEN_{factor}/{img_name}', img_result)