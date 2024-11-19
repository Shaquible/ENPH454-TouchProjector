# This class defines the cropped area of the screen that will be used by the two cameras as a region of interest
import numpy as np
from screeninfo import get_monitors
from triangulationCharuco import Camera, Triangulation


def crop(xy_uv, cam1: Camera, cam2: Camera, Overshoot: float) ->tuple[list[int], list[int]]:
    # Pass the real space to projector space matrix and the two camera objects. Also pass the desired overshoot percentage

    # Finds the full pixel resolution
    for m in get_monitors():
        xMax = m.width
        yMax = m.height

    # matrix for the world coordinates (x,y,z) given the uv values of the screen. Z is appended on as zero.
    A = np.linalg.inv(xy_uv)
    mat1 = cam1.irProjection
    mat2 = cam2.irProjection

    # defines the desired points that are an overshoot of the screen size
    Mag1 = np.sqrt(xMax**2+yMax**2)*(1+Overshoot)
    Mag2 = np.sqrt((xMax*Overshoot)**2+(yMax*(1+Overshoot)**2))
    
    points = [(-1, 0, xMax*Overshoot), 
              (1, 0, xMax*(1+Overshoot)), 
              (xMax*(1+Overshoot)/Mag1, yMax*(1+Overshoot)/Mag1, Mag1),
              (-xMax*Overshoot/Mag2, yMax*(1+Overshoot)/Mag2, Mag2)]
    pointsXYW = np.zeros((4, 3))
    pointsXYZW = np.zeros((4, 4))
    pointsCam1 = np.zeros((4, 3))
    pointsCam2 = np.zeros((4, 3))

    for i in range(4):
        pointsXYW[i] = np.matmul(A, points[i])
        pointsXYZW[i] = [pointsXYW[i][0],pointsXYW[i][1], 0, pointsXYW[i][2]]
        pointsCam1[i] = np.matmul(mat1, pointsXYZW[i])
        pointsCam2[i] = np.matmul(mat2, pointsXYZW[i])

    # Gets the corner point for camera 1
    u1C1 = int(min(pointsCam1[:, 0]))
    v1C1 = int(min(pointsCam1[:, 1]))
    # Gets the corner point for camera 2
    u1C2 = int(min(pointsCam2[:, 0]))
    v1C2 = int(min(pointsCam2[:, 1]))
    # Gets the range for camera 1
    width_uC1 = int(max(pointsCam1[:, 0]) - u1C1)
    width_vC1 = int(max(pointsCam1[:, 1]) - v1C1)
    # gets range for camera 2
    width_uC2 = int(max(pointsCam2[:, 0]) - u1C2)
    width_vC2 = int(max(pointsCam2[:, 1]) - v1C2)

    return [u1C1, v1C1, width_uC1, width_vC1], [u1C2, v1C2, width_uC2, width_vC2]


if __name__ == "__main__":
    npfile = np.load("cameraIntrinsics/Cam1Vis.npz")
    mtx1 = npfile["mtx"]
    dist1 = npfile["dist"]
    npfile = np.load("cameraIntrinsics/Cam2Vis.npz")
    mtx2 = npfile["mtx"]
    dist2 = npfile["dist"]
    npfile = np.load("cameraIntrinsics/Cam1IR.npz")
    mtx1IR = npfile["mtx"]
    dist1IR = npfile["dist"]
    npfile = np.load("cameraIntrinsics/Cam2IR.npz")
    mtx2IR = npfile["mtx"]
    dist2IR = npfile["dist"]
    
    tri = Triangulation(Camera(mtx1IR, dist1IR, mtx1, dist1),
                        Camera(mtx2IR, dist2IR, mtx2, dist2))
    
    xy_to_uv_mat = np.identity(3)
    
    print(crop(xy_to_uv_mat, tri.cam1, tri.cam2, 0.1))