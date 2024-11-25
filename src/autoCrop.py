# This class defines the cropped area of the screen that will be used by the two cameras as a region of interest
import numpy as np
from screeninfo import get_monitors
from triangulationCharuco import Camera, Triangulation

def crop(xy_uv, cam1: Camera, cam2: Camera, Overshoot: float) ->tuple[list[int], list[int]]:
#def crop(xy_uv, cam1, cam2, Overshoot: float) ->tuple[list[int], list[int]]:
    # Pass the real space to projector space matrix and the two camera objects. Also pass the desired overshoot percentage

    # Finds the full pixel resolution
    for m in get_monitors():
        xMax = m.width
        yMax = m.height
    xMax = 3840
    yMax = 2160
    # matrix for the world coordinates (x,y,z) given the uv values of the screen. Z is appended on as zero.
    A = np.linalg.inv(xy_uv)
    #.irProjection if from a camera object
    mat1 = cam1.irProjection
    mat2 = cam2.irProjection

    # defines the desired points that are an overshoot of the screen size
    Mag1 = np.sqrt(xMax**2+yMax**2)*(1+Overshoot)
    Mag2 = np.sqrt((xMax*Overshoot)**2+(yMax*(1+Overshoot)**2))
    
    points = np.array([(-xMax*Overshoot, 0, 1), 
              (xMax*(1+Overshoot), 0, 1), 
              (xMax*(1+Overshoot), yMax*(1+Overshoot), 1),
              (-xMax*Overshoot, yMax*(1+Overshoot), 1)])
    print(points)
    pointsXYW = np.zeros((4, 3))
    pointsXYZW = np.zeros((4, 4))
    pointsCam1 = np.zeros((4, 3))
    pointsCam2 = np.zeros((4, 3))

    for i in range(4):
        pointsXYW[i] = np.matmul(A, points[i])
        pointsXYW[i] = pointsXYW[i]/pointsXYW[i][2]
        pointsXYZW[i] = [pointsXYW[i][0],pointsXYW[i][1], 0, pointsXYW[i][2]]
        pointsCam1[i] = np.matmul(mat1, pointsXYZW[i])
        pointsCam1[i] = pointsCam1[i] / pointsCam1[i][2]
        pointsCam2[i] = np.matmul(mat2, pointsXYZW[i])
        pointsCam2[i] = pointsCam2[i] / pointsCam2[i][2]
    print(pointsXYW)
    print(pointsCam1)
    

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
    
    npfile = np.load("src/johnDebug.npz")
    print(npfile.files)
    cam1 = npfile["cam1Pose"]
    cam2 = npfile["cam2Pose"]
    xy_to_uv_mat = npfile["xy_to_uv_mat"]
    
    # xy_to_uv_mat = np.array([[ 6.47527889e+03,  2.10521119e+02,  6.10772855e+02],
 #[ 1.57601980e+01,  6.10275228e+03,  6.16671078e+02],
 #[-5.64507939e-04,  1.12740223e-01,  1.00000000e+00]])
    
    print(crop(xy_to_uv_mat, cam1, cam2, 0.1))