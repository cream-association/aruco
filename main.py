import numpy as np
import cv2
import logging
from utils import (estimatePoseSingleMarkers, get_center_from_tag_corners,
                   get_distance_from_center, compute_x_borders)


logging.basicConfig()
LOGGER = logging.getLogger("aruco")
LOGGER.setLevel(logging.ERROR)

GUI_available = False

aruco_tags_to_detect = [36, 13]  # Tag number of delicate and tough flowers resp


if __name__ == "__main__":
    LOGGER.info("Starting video capture ...")
    vid = cv2.VideoCapture(0)

    # detecting aruco
    LOGGER.info("Loading aruco dict")
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    while True:
        LOGGER.info("reading next frame")
        ret, frame = vid.read()

        frame_height, frame_width, _ = frame.shape

        LOGGER.info("detect ArUco markers in the input frame")
        (corners, ids, rejected) = detector.detectMarkers(frame)

        LOGGER.info("verify *at least* one ArUco marker was detected")

        corners_to_consider = []

        if len(corners) > 0:
            #LOGGER.debug("flatten the ArUco IDs list")
            ids = ids.flatten()

            LOGGER.debug("loop over the detected ArUCo corners")
            for markerCorner, markerID in zip(corners, ids):
                LOGGER.info(
                        "extract the marker corners (which are always returned "
                        "in top-left, top-right, bottom-right, and bottom-left "
                        "order"
                        )

                to_consider = False
                # In some cases (Why?) markerID is not a int, but a list of int
                try:
                        to_consider = markerID in aruco_tags_to_detect

                except Exception:
                        pass

                if to_consider == True:
                        tag_corners_coord = markerCorner.reshape((4, 2))
                        c = tag_corners_coord, markerID
                        corners_to_consider.append(c)


            LOGGER.info(f"Number of relevants tags: {len(corners_to_consider)}")
            if len(corners_to_consider) > 0:
                    LOGGER.info("Looking for the tag closest to the middle of the camera")
                    tags_centers = [get_center_from_tag_corners(corners_coord) for corners_coord, _ in corners_to_consider]
                    distances_from_center = [get_distance_from_center(tag_center, (frame_width, frame_height)) for tag_center in tags_centers]
                    index_to_focus = np.argmin(distances_from_center)

                    rvec, tvec, _ = estimatePoseSingleMarkers([corners_to_consider[index_to_focus][0]], 0.01)
                    dist = np.linalg.norm(tvec) * 100
                    if dist < 10:
                            LOGGER.info("Do not need to move anymore")
                    elif dist < 20:
                            cX, cY = tags_centers[index_to_focus]
                            borders = list(compute_x_borders(
                                cX, dist, frame_width, 10))

                            x_left_border, x_right_border = min(borders), max(borders)
                            LOGGER.info(f"BORDER: {x_left_border, x_right_border}")


                            print("borders", x_left_border, x_right_border)
                            if cX < x_left_border - 20:
                                    LOGGER.info("GO LEFT")
                            elif x_right_border + 20 < cX:
                                LOGGER.info("GO RIGHT")

                            cv2.line(frame,
                             (int(x_left_border), 0),
                             (int(x_left_border), frame_height),
                             (0, 255, 0), 2)
                            cv2.line(frame,
                             (int(x_right_border), 0),
                             (int(x_right_border), frame_height),
                             (0, 255, 0), 2)
                    else:
                        LOGGER.info("GO STRAIGHT")

        if GUI_available == True:
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                        LOGGER.info("Exit key pressed, exiting")
                        break

    vid.release()
    cv2.destroyAllWindows()

    LOGGER.info("video release and window destroyed. Bye.")
