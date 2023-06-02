# getting skeleton for lobster
import numpy as np


# get the v position
# return the list of locations for lobster

def get_skeleton(v):
    # center 1
    c1 = np.array([v[1], v[2], v[3], v[4], v[5]]).mean(axis=0)

    # left 1
    l1 = np.array([v[22], v[23]]).mean(axis=0)

    # left 2
    l2 = np.array([v[25], v[26]]).mean(axis=0)

    # right 1
    r1 = np.array([v[12], v[13]]).mean(axis=0)

    # right 2
    r2 = np.array([v[15], v[16]]).mean(axis=0)

    # center 2
    c2 = np.array([v[17], v[18], v[19]]).mean(axis=0)

    # center 3
    c3 = np.array([v[34], v[36], v[37]]).mean(axis=0)

    # center 4
    c4 = np.array([v[38], v[41], v[43]]).mean(axis=0)

    return np.array([c1, c2, c3, c4, l1, l2, r1, r2])


def get_skeleton_edge():
    l = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [0, 6], [6, 7]]
    return np.array(l)
