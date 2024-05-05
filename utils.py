import numpy as np
import pickle
import cv2


with open("calibration.pckl", "rb") as f:
    data = pickle.load(f)
    cMat = data[0]
    dcoeff = data[1]


def estimatePoseSingleMarkers(corners, marker_size, mtx=cMat,distortion=dcoeff):
    """
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    """
    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )
    trash = []
    rvecs = []
    tvecs = []

    for c in corners:
        nada, R, t = cv2.solvePnP(
            marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE
        )
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash


def compute_x_borders(marker_x, dist, frame_width, max_dist):
    return compute_left_border(
        marker_x, dist, frame_width, max_dist
        ),  compute_right_border(
        marker_x, dist, frame_width, max_dist
        )


def compute_right_border(marker_x, dist, frame_width, max_dist):
    return (max_dist / dist) * 0.9 * frame_width


def compute_left_border(marker_x, dist, frame_width, max_dist):
    return frame_width - compute_right_border(marker_x,
                                              dist,
                                              frame_width,
                                              max_dist)



