def createPlane(P1,P2, P3):
    v1 = P2 - P1
    v2 = P3 - P1

    # Calculate the normal vector using the cross product
    v3 = np.cross(v1, v2)



    # Normalize the normal vector (optional)
    v1Norm = v1 / np.linalg.norm(v1)
    v2Norm = v2 / np.linalg.norm(v2)
    v3Norm = v3 / np.linalg.norm(v3)

    return(v1Norm,v2Norm,v3Norm)

def transform(center, v1,v2,v3):

    transform = [[v1[0],v1[1], v1[2], -center[0] ], [v2[0],v2[1], v2[2], -center[1] ],[v3[0],v3[1], v3[2], -center[2] ],[0,0, 0, 1 ]]
    return transform
