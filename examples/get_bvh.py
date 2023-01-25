from utils.bvh_skeleton.bvh import Bvh
from utils.skeleton import get_skeleton_edge
from utils.visualizer import showFrames
import numpy as np

anim = Bvh()
anim.parse_file('./output/result_fixed.bvh')
# get position at frame 11
positions, rotations = anim.all_frame_poses()

# get name of joints in order of appearance in array
joint_names = anim.joint_names()

print(positions.shape)
x = positions[:, :, 0]
z = positions[:, :, 1]
y = positions[:, :, 2]

arr1 = np.stack([x, y, z], axis=2)

# anim.plot_all_frames()
# edge = get_skeleton_edge()
edge = get_skeleton_edge()

showFrames(arr1, edge)
