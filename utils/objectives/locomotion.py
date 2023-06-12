import numpy as np

from utils.geometry import getFrontDirection
from utils.objectives.objective import Objective
from utils.truss import Truss


class Locomotion(Objective):

    def __init__(self, truss: Truss):
        self.truss = truss

    def execute(self):
        pass


class MoveForward(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        dx = (vs[-1].mean(0) - vs[0].mean(0))[0]
        velX = dx
        return velX

class MoveLeft(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)
    def execute(self):
        vs = self.truss.vs
        left_claw  = np.array([10 ,21, 22, 23, 24, 25, 26, 30, 31, 32],dtype = int)
        dx = (vs[-1][left_claw].mean(0) - vs[0][left_claw].mean(0))[0]
        velX = -dx
        return velX

class MoveRight(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)
    def execute(self):
        vs = self.truss.vs
        right_claw  = np.array([9 ,11 ,12 ,13 ,14, 15, 16, 27, 28, 29],dtype = int)
        dx = (vs[-1][right_claw].mean(0) - vs[0][right_claw].mean(0))[0]
        velX = dx
        return velX

class MoveUp(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)
    def execute(self):
        vs = self.truss.vs
        head  = np.array([ 0,  1  ,2,  3 , 4,  5,  6,  7 , 8, 17 ,18, 19],dtype = int)
        dy = (vs[-1][head].mean(0) - vs[0][head].mean(0))[1]
        velY = dy
        return velY    

class MoveDown(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)
    def execute(self):
        vs = self.truss.vs
        body  = np.array([20, 33, 34, 35, 36, 37 ,38, 39, 40, 41 ,42, 43, 44],dtype = int)
        dy = (vs[-1][body].mean(0) - vs[0][body].mean(0))[1]
        velY = -dy
        return velY
    

class MoveUpward(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        dy = (vs[-1].mean(0) - vs[0].mean(0))[1]
        velY= dy
        return velY

class FaceForward(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        # 2d direction
        vs = self.truss.vs
        vecFront = getFrontDirection(vs[0], vs[-1])  # unit Vector
        assert (abs((vecFront ** 2).sum() - 1) < 1e-6)
        vecX = np.array([1, 0, 0])
        alignment = (vecFront * vecX).sum()
        return alignment


class TurnLeft(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        vecFront = getFrontDirection(vs[0], vs[-1])  # unit Vector
        assert (abs((vecFront ** 2).sum() - 1) < 1e-6)
        vecL = np.array([0, 1, 0])
        alignment = (vecFront * vecL).sum()
        return alignment


class TurnRight(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        vecFront = getFrontDirection(vs[0], vs[-1])  # unit Vector
        assert (abs((vecFront ** 2).sum() - 1) < 1e-6)
        vecL = np.array([0, -1, 0])
        alignment = (vecFront * vecL).sum()
        return alignment


class LowerBodyMax(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        return -vs[:, :, 2].max()


class LowerBodyMean(Locomotion):
    def __init__(self, truss: Truss):
        super().__init__(truss)

    def execute(self):
        vs = self.truss.vs
        return -vs[:, :, 2].max(1).mean()
