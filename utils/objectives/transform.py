import numpy as np
import igl as igl
from utils.mesh import Mesh

from utils.objectives.objective import Objective
from utils.truss import Truss
from utils.geometry import boundingBox


class Transform(Objective):

    def __init__(self, truss: Truss, mesh: Mesh):
        self.truss = truss
        self.mesh = mesh

    def execute(self):
        pass


class KeyPointsAlign(Transform):

    def __init__(self, truss: Truss, mesh: Mesh):
        super().__init__(truss, mesh)

    def execute(self):
        mesh_keypoints: np.ndarray = self.mesh.v[self.truss.indices]

        v_keypoints = self.truss.vs[4, self.truss.indices] #4 is the last frame of the first action.

        # print(v_keypoints.shape)
        # print(mesh_keypoints.shape)
        assert (v_keypoints.shape == mesh_keypoints.shape)
        return -np.sqrt(((v_keypoints - mesh_keypoints) ** 2).sum(1)).mean()
    
#make sure the last frame is aligned with the original frame to prevent buckling
class OriginAlign(Transform):

    def __init__(self, truss: Truss, mesh: Mesh):
        super().__init__(truss, mesh)

    def execute(self):
        # mesh_keypoints: np.ndarray = self.mesh.keyPoints

        # v_keypoints = self.truss.vs[-1, self.truss.indices]
        # assert (v_keypoints.shape == mesh_keypoints.shape)
        #to calculate last frame - first frame is 0.
        return -np.sqrt(((self.truss.vs[-1,:] - self.truss.vs[0,:]) ** 2).sum(1)).mean()
    

class SideAlign(Transform):

    def __init__(self, truss: Truss, mesh: Mesh):
        super().__init__(truss, mesh)

    def execute(self):
        #[9, 22, 23, 52,53,54, 55]
        #keep the other points the same ..
        mesh_keypoints: np.ndarray = self.mesh.keyPoints

        v_keypoints = self.truss.vs[-1, self.truss.indices]

        v_initpoints = self.truss.vs[0, self.truss.indices]

        assert (v_keypoints.shape == mesh_keypoints.shape)

        sideScore= -np.sqrt(((v_keypoints[0:3] - mesh_keypoints[0:3]) ** 2).sum(1)).mean()
        frontScore = -np.sqrt(((v_keypoints[3:] - v_initpoints[3:]) ** 2).sum(1)).mean()
        return  sideScore + frontScore
    
class FrontAlign(Transform):

    def __init__(self, truss: Truss, mesh: Mesh):
        super().__init__(truss, mesh)

    def execute(self):
        #[9, 22, 23, 52,53,54, 55]
        mesh_keypoints: np.ndarray = self.mesh.keyPoints

        v_keypoints = self.truss.vs[-1, self.truss.indices]
        v_initpoints = self.truss.vs[0, self.truss.indices]

        assert (v_keypoints.shape == mesh_keypoints.shape)

        #side should stay the same
        sideScore= -np.sqrt(((v_keypoints[0:3] - v_initpoints[0:3]) ** 2).sum(1)).mean()

        #front should align with target mesh keypoints
        frontScore = -np.sqrt(((v_keypoints[3:] -mesh_keypoints[3:]) ** 2).sum(1)).mean()
    
        assert (v_keypoints.shape == mesh_keypoints.shape)
        return sideScore + frontScore


class SurfaceAlign(Transform):
    def __init__(self, truss: Truss, mesh: Mesh):
        super().__init__(truss, mesh)

    def execute(self):
        self.mesh.affine(boundingBox(self.truss.vs[-1]))
        # apply affine transformation to the mesh to have equal ratio with the truss
        surface: np.ndarray = self.mesh.surface
        v_mesh: np.ndarray = self.mesh.v
        v_points = self.truss.vs[-1]
        dis, _, _ = igl.point_mesh_squared_distance(v_points, v_mesh, surface)
        return - (sum(dis) / len(dis))
