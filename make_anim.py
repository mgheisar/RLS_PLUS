import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import cv2
import numpy as np

fig = plt.figure()  # make figure
ims = []
strr = "sandeep-project-"
for i in range(1000):
    np_image = cv2.imread("input/project/gif/" + strr + "%d.png" % i)
    ims.append(np_image)


# im = cv2.imshow("image[0]", ims[0])
#
# def updatefig(j):
#     # set the data in the axesimage object
#     im.set_array(ims[j])
#     print(j)
#     # return the artists set
#     return [im]
# # kick off the animation
# anim = animation.FuncAnimation(fig, updatefig, frames=range(len(ims)), interval=50, blit=True)
# anim.save('animation_face_projection_1.gif', writer='imagemagick', fps=4)

# import glob
# from PIL import Image
#
# # filepaths
# fp_in = "runs/000005_4/HR/*.png"
# fp_out = "anim.gif"
# img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
# img.save(fp=fp_out, format='GIF', append_images=imgs,
#          save_all=True, duration=500, loop=0)


def write_video(file_path, frames, fps=30):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """

    w, h, cc = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(frame)

    writer.release()


write_video("anim_" + strr + ".mp4", ims)
