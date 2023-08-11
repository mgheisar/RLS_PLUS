import replicate
from PIL import Image
import numpy as np
import os
import urllib.request
import glob
from pathlib import Path

os.environ["REPLICATE_API_TOKEN"] = "b03a484cc391ceb4fc4ce1112dd38b94a419685d"
client = replicate.Client(api_token="b03a484cc391ceb4fc4ce1112dd38b94a419685d")
# Load the model
model = replicate.models.get("xinntao/gfpgan")
version = model.versions.get("6129309904ce4debfde78de5c209bce0022af40e197e132f08be8ccce3050393")
factor = 16
img_list = sorted(glob.glob(f"input/project/resLR_{factor}x/*.jpg"))
for img_file in img_list:
    inputs = {
        # Input
        'img': open(img_file, "rb"),
        'version': "v1.2",
        # Rescaling factor
        'scale': factor,
    }
    img_name = (Path(img_file).stem.split('_')[0] + ".jpg")
    # Run the model
    output = version.predict(**inputs)
    urllib.request.urlretrieve(output, f'input/project/resSR/GFPGAN_{factor}/{img_name}')
