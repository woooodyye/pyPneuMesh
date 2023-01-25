from . import math3d
from . import bvh_helper

import numpy as np
from pprint import pprint


class LobsterSkeleton(object):

    def __init__(self):
        self.root = 'Spine'
        self.keypoint2index = {
            'Spine': 0,
            'Spine1': 1,
            'Spine2': 2,
            'Spine3': 3,
            'LeftArm': 4,
            'LeftClaw': 5,
            'RightArm': 6,
            'RightClaw': 7
        }
        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}
        self.keypoint_num = len(self.keypoint2index)

        self.children = {
            'Spine': ['LeftArm', 'Spine1', 'RightArm'],
            'Spine1': ['Spine2'],
            'Spine2': ['Spine3'],
            'Spine3': [],
            'LeftArm': ['LeftClaw'],
            'LeftClaw': [],
            'RightArm': ['RightClaw'],
            'RightClaw': []
        }
        self.parent = {self.root: None}
        for parent, children in self.children.items():
            for child in children:
                self.parent[child] = parent

        self.left_joints = [
            joint for joint in self.keypoint2index
            if 'Left' in joint
        ]
        self.right_joints = [
            joint for joint in self.keypoint2index
            if 'Right' in joint
        ]

        # T-pose
        self.initial_directions = {
            'Spine': [0, 0, 0],
            'Spine1': [0, 1, 0],
            'Spine2': [0, 1, 0],
            'Spine3': [0, 1, 0],
            'LeftArm': [1, 0, 0],
            'LeftClaw': [1, 0, 0],
            'RightArm': [-1, 0, 0],
            'RightClaw': [-1, 0, 0]
        }

    def get_initial_offset(self, poses_3d):
        # TODO: RANSAC
        bone_lens = {self.root: [0]}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            p_idx = self.keypoint2index[parent]
            p_name = parent
            while p_idx == -1:
                # find real parent
                p_name = self.parent[p_name]
                p_idx = self.keypoint2index[p_name]
            for child in self.children[parent]:
                stack.append(child)

                if self.keypoint2index[child] == -1:
                    bone_lens[child] = [0.1]
                else:
                    c_idx = self.keypoint2index[child]
                    bone_lens[child] = np.linalg.norm(
                        poses_3d[:, p_idx] - poses_3d[:, c_idx],
                        axis=1
                    )

        bone_len = {}
        for joint in self.keypoint2index:
            if 'Left' in joint or 'Right' in joint:
                base_name = joint.replace('Left', '').replace('Right', '')
                left_len = np.mean(bone_lens['Left' + base_name])
                right_len = np.mean(bone_lens['Right' + base_name])
                bone_len[joint] = (left_len + right_len) / 2
            else:
                bone_len[joint] = np.mean(bone_lens[joint])

        initial_offset = {}
        for joint, direction in self.initial_directions.items():
            direction = np.array(direction) / max(np.linalg.norm(direction), 1e-12)
            initial_offset[joint] = direction * bone_len[joint]

        return initial_offset

    def get_bvh_header(self, poses_3d):
        initial_offset = self.get_initial_offset(poses_3d)

        nodes = {}
        for joint in self.keypoint2index:
            is_root = joint == self.root
            is_end_site = 'EndSite' in joint
            nodes[joint] = bvh_helper.BvhNode(
                name=joint,
                offset=initial_offset[joint],
                rotation_order='zxy' if not is_end_site else '',
                is_root=is_root,
                is_end_site=is_end_site,
            )
        for joint, children in self.children.items():
            nodes[joint].children = [nodes[child] for child in children]
            for child in children:
                nodes[child].parent = nodes[joint]

        header = bvh_helper.BvhHeader(root=nodes[self.root], nodes=nodes)
        return header

    def pose2euler(self, pose, header):
        channel = []
        quats = {}
        eulers = {}
        stack = [header.root]
        while stack:
            node = stack.pop()
            joint = node.name
            joint_idx = self.keypoint2index[joint]

            if node.is_root:
                channel.extend(pose[joint_idx])

            index = self.keypoint2index
            order = None
            if joint == 'Spine':
                x_dir = pose[index['LeftArm']] - pose[index['RightArm']]
                y_dir = pose[index['Spine1']] - pose[joint_idx]
                z_dir = None
                order = 'yzx'
            elif joint == 'Spine1':
                x_dir = pose[index['LeftArm']] - pose[index['RightArm']]
                y_dir = pose[index['Spine2']] - pose[index['Spine']]
                z_dir = None
                order = 'yzx'
            elif joint == 'Spine2':
                x_dir = pose[index['LeftArm']] - pose[index['RightArm']]
                y_dir = pose[index['Spine3']] - pose[index['Spine1']]
                z_dir = None
                order = 'yzx'
            elif joint == 'Spine3':
                x_dir = pose[index['LeftArm']] - pose[index['RightArm']]
                y_dir = pose[joint_idx] - pose[index['Spine2']]
                z_dir = None
                order = 'yzx'
            elif joint == 'LeftArm':
                x_dir = pose[index['LeftClaw']] - pose[joint_idx]
                y_dir = pose[index['LeftClaw']]
                z_dir = None
                order = 'xzy'
            elif joint == 'RightArm':
                x_dir = pose[joint_idx] - pose[index['RightClaw']]
                y_dir = pose[index['RightClaw']]
                z_dir = None
                order = 'xzy'

            if order:
                dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                quats[joint] = quats[self.parent[joint]].copy()

            local_quat = quats[joint].copy()
            if node.parent:
                local_quat = math3d.quat_divide(
                    q=quats[joint], r=quats[node.parent.name]
                )

            euler = math3d.quat2euler(
                q=local_quat, order=node.rotation_order
            )
            euler = np.rad2deg(euler)
            eulers[joint] = euler
            channel.extend(euler)

            for child in node.children[::-1]:
                if not child.is_end_site:
                    stack.append(child)

        return channel

    def poses2bvh(self, poses_3d, header=None, output_file=None):
        if not header:
            header = self.get_bvh_header(poses_3d)

        channels = []
        for frame, pose in enumerate(poses_3d):
            channels.append(self.pose2euler(pose, header))

        if output_file:
            bvh_helper.write_bvh(output_file, header, channels)

        return channels, header
