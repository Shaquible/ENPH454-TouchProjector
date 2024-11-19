# This class defines the cropped area of the screen that will be used by the two cameras as a region of interest
import numpy as np
from screeninfo import get_monitors
from triangulationCharuco import Camera


def crop(xy_uv, cam1: Camera, cam2: Camera, Overshoot: float) ->tuple[list[int], list[int]]:
    # Pass the real space to projector space matrix and the two camera objects. Also pass the desired overshoot percentage

    # Finds the full pixel resolution
    for m in get_monitors():
        xMax = m.width
        yMax = m.height

    # matrix for the world coordinates (x,y,z) given the uv values of the screen. Z is appended on as zero.
    A = np.append(np.linalg.inv(xy_uv), 0, 1)
    mat1 = cam1.irProjection
    mat2 = cam2.irProjection

    # defines the desired points that are an overshoot of the screen size
    points = [(-xMax*Overshoot, 0), (xMax*(1+Overshoot), 0), (xMax*(1+Overshoot),
                                                              yMax*(1+Overshoot)), (-xMax*Overshoot, yMax*(1+Overshoot))]
    pointsXYZ = np.zeros(4, 3)
    pointsCam1 = np.zeros(4, 2)
    pointsCam2 = np.zeros(4, 2)

    for i in range(4):
        pointsXYZ[i] = np.matmul(A, points[i])
        pointsCam1[i] = np.matmul(mat1, pointsXYZ[i])
        pointsCam2[i] = np.matmul(mat2, pointsXYZ[i])

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
