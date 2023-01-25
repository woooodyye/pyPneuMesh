import numpy as np
from pyPneuMesh.utils import readNpy
from pyPneuMesh.geometry import getFrontDirection


def objMoveForward(vs: np.ndarray, vEnergys: np.ndarray):
    dx = (vs[-1].mean(0) - vs[0].mean(0))[0]
    velX = dx
    return velX


def objFaceForward(vs: np.ndarray, vEnergys: np.ndarray):
    # 2d direction
    vecFront = getFrontDirection(vs[0], vs[-1])  # unit Vector
    assert (abs((vecFront ** 2).sum() - 1) < 1e-6)
    vecX = np.array([1, 0, 0])
    alignment = (vecFront * vecX).sum()
    return alignment


def objTurnLeft(vs: np.ndarray, vEnergys: np.ndarray):
    vecFront = getFrontDirection(vs[0], vs[-1])  # unit Vector
    assert (abs((vecFront ** 2).sum() - 1) < 1e-6)
    vecL = np.array([0, 1, 0])
    alignment = (vecFront * vecL).sum()
    return alignment


def objLowerBody(vs: np.ndarray, vEnergys: np.ndarray):
    zMax = vs[:, :, 2].max()
    return -zMax


def objTableTilt(vs: np.ndarray, vEnergys: np.ndarray):
    v45 = vs[-1, 45]
    v46 = vs[-1, 46]
    v47 = vs[-1, 47]
    v48 = vs[-1, 48]

    vec0 = v45 - v46
    vec1 = v47 - v48

    vecPerpendicular = np.cross(vec0, vec1)
    unitVecPerpendicular = vecPerpendicular / np.linalg.norm(vecPerpendicular)
    horizontal = np.linalg.norm(unitVecPerpendicular[:2])
    vertical = unitVecPerpendicular[2]
    unitVec = np.array([horizontal, vertical])
    unitVec45 = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])

    return np.dot(unitVec, unitVec45)


def objTableNoTilt(vs: np.ndarray, vEnergys: np.ndarray):
    # minimize the z axis difference of the two pairs of the top four nodes
    return - abs(vs[-1, 45, 2] - vs[-1, 46, 2]) - abs(vs[-1, 47, 2] - vs[-1, 48, 2])


def objTableAlwaysNoTilt(vs: np.ndarray, vEnergys: np.ndarray):
    # minimize the z axis difference of the two pairs of the top four nodes
    return - ((vs[:, 45, 2] - vs[:, 46, 2]) ** 2).sum() - ((vs[:, 47, 2] - vs[:, 48, 2]) ** 2).sum()


def objTurnRight(vs: np.ndarray, vEnergys: np.ndarray):
    vecFront = getFrontDirection(vs[0], vs[-1])  # unit Vector
    assert (abs((vecFront ** 2).sum() - 1) < 1e-6)
    vecL = np.array([0, -1, 0])
    alignment = (vecFront * vecL).sum()
    return alignment


def objTableHigh(vs: np.ndarray, vEnergys: np.ndarray):
    indicesTop = [0, 1, 2, 3]
    zTarget = 5

    vs = vs[:, indicesTop]
    nFrames = 5
    interval = len(vs) / nFrames
    vs = vs[0::interval] + [vs[-1]]
    zMean = vs[:, :, 2].mean()

    return abs(zMean - zTarget)


def objGrabLobster(vs: np.ndarray):
    return -np.sqrt(((vs[:, 32] - vs[:, 29]) ** 2).sum(1).max())


def objLowerBodyMax(vs: np.ndarray, vEnergys: np.ndarray):
    return -vs[:, :, 2].max()


def objLowerBodyMean(vs: np.ndarray, vEnergys: np.ndarray):
    return -vs[:, :, 2].max(1).mean()


def objMinEnergy(vs: np.ndarray, vEnergys: np.ndarray):
    totalE = vEnergys.sum()
    dx = (vs[-1].mean(0) - vs[0].mean(0))[0]

    # E is always positive, wants to minimize Energy.
    return - (totalE / dx)


def objHelmetAlign(vs: np.ndarray):
    indices = []

    # import mesh here
    # io runtime
    # mesh npy data
    # and then just compute diff
    # this do be bad bc it's specific file  to be read
    meshParam = readNpy('scripts/testLobster/lobster/lobster.trussparam.npy')

    # return -np.sqrt(((vs[:, 32] - vs[:, 29]) ** 2).sum(1).max())
