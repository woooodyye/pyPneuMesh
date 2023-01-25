import json
import numpy as np

from utils.bvh_skeleton import lobster_skeleton

skeletonDir = 'output/frames_.json'

with open(skeletonDir) as ifile:
    content = ifile.read()
data = json.loads(content)

vs = np.array(data['vs'])

bvh_file = 'output/lobster.bvh'
lobster_skel = lobster_skeleton.LobsterSkeleton()
channels, header = lobster_skel.poses2bvh(vs, output_file=bvh_file)
