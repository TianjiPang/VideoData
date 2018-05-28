import numpy as np
import os
import flow_utils
new_width = 224
new_height = 224
num = 25
path = r"./flow_v_CricketShot_g04_c01_resized"
filenames = os.listdir(path)
filenames.sort(key=lambda x:int(x[:-4]))
flows = []
normed_flows = []
for i in range(len(filenames) / num):
    my_filename = filenames[num * i:num * (i + 1)]
    for filename in list(my_filename):
        flow = flow_utils.readFlow(os.path.join(path, filename))
        x = flow[:, :, 0]
        y = flow[:, :, 1]
        normed_x = ((x - x.min()) / (x.max() - x.min()) - 0.5) * 2
        normed_y = ((y - y.min()) / (y.max() - y.min()) - 0.5) * 2
        width, height, channel = flow.shape
        normed_flow = np.zeros((width, height, 2))
        normed_flow[:, :, 0] = normed_x
        normed_flow[:, :, 1] = normed_y
        left = (width - new_width) / 2
        right = (width + new_width) / 2
        top = (height - new_height) / 2
        bottom = (height + new_height) / 2
        croped_flow = flow[left:right, top:bottom, :]
        croped_normed_flow = normed_flow[left:right, top:bottom, :]
        flows.append(np.array(croped_flow))
        normed_flows.append(np.array(croped_normed_flow))
    my_flows = np.array([[np.array(frame) for frame in flows]])
    my_normed_flows = np.array([[np.array(frame) for frame in normed_flows]])
    seq1 = ['./TBCresized_flo/1s/', str(i + 1).zfill(4), '.npy']
    seq2 = ['./TBCresizedNormed_flo/1s/', str(i + 1).zfill(4), '.npy']
    np.save(''.join(seq1), my_flows)
    np.save(''.join(seq2), my_normed_flows)