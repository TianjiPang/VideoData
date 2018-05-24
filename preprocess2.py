import numpy as np
import os
import cv2
num = 125
new_width = 224
new_height = 224
path = r"/disk3/tianji/videoFea/TBCresized"
filenames = os.listdir(path)
filenames.sort(key=lambda x:int(x[:-4]))
def my_resize(path, filenames):
    for filename in list(filenames):
        print(filename)
        copy_frame = cv2.imread(os.path.join(path,filename))
        g, r, b = cv2.split(copy_frame)
        g = g.astype(np.float32)
        r = r.astype(np.float32)
        b = b.astype(np.float32)
        rescaled_r = ((r - r.min()) / (r.max() - r.min()) - 0.5) * 2
        rescaled_g = ((g - g.min()) / (g.max() - g.min()) - 0.5) * 2
        rescaled_b = ((b - b.min()) / (b.max() - b.min()) - 0.5) * 2
        rescaled_image = cv2.merge([rescaled_g, rescaled_r, rescaled_b])
        width, height, color = rescaled_image.shape
        left = (width - new_width) / 2
        right = (width + new_width) / 2
        top = (height - new_height) / 2
        bottom = (height + new_height) / 2
        croped_frame = rescaled_image[left:right, top:bottom, :]
        yield croped_frame

for i in range(len(filenames)/num):
    my_filename = filenames[num*i:num*(i+1)]
    copy_frame = my_resize(path, my_filename)
    frames = np.array([[np.array(frame) for frame in copy_frame]])
    seq = ['./TBC/5s/', str(i+1).zfill(4), '.npy']
    np.save(''.join(seq), frames)
