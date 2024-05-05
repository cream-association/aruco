import numpy as np
import cv2
import logging
from utils import estimatePoseSingleMarkers, compute_x_borders

logging.basicConfig()
LOGGER = logging.getLogger("aruco")
LOGGER.setLevel(logging.DEBUG)


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
        if len(corners) > 0:
            LOGGER.debug("flatten the ArUco IDs list")
            ids = ids.flatten()

            LOGGER.debug("loop over the detected ArUCo corners")
            for markerCorner, markerID in zip(corners, ids):
                LOGGER.info(
                        "extract the marker corners (which are always returned "
                        "in top-left, top-right, bottom-right, and bottom-left "
                        "order"
                        )
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                LOGGER.info("convert each of the (x, y)-coordinate pairs to integers")
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                LOGGER.info("draw the bounding box of the ArUCo detection")
                cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

                LOGGER.info("Estimating aruco position")
                rvec, tvec, _ = estimatePoseSingleMarkers(
                        markerCorner, 0.01)

                dist = np.linalg.norm(tvec) * 100

                LOGGER.info("draw the ArUco marker ID on the frame")
                cv2.putText(
                        frame,
                        f"{markerID} - Distance: {dist:.2f} cm",
                        (topLeft[0], topLeft[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                        )

                LOGGER.info(
                        "compute and draw the center (x, y)-coordinates of the"
                        " ArUco marker"
                        )
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

                LOGGER.info("draw the ArUco marker ID on the frame")
                cv2.putText(
                        frame,
                        str(markerID),
                        (topLeft[0], topLeft[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                        )

                if dist < 10:
                    LOGGER.info("Do not need to move anymore")
                elif dist < 20:
                    x_left_border, x_right_border = compute_x_borders(
                            cX, dist, frame_width, 10)


                    # Drawing border lines
                    cv2.line(frame,
                             (int(x_left_border), 0),
                             (int(x_left_border), frame_height),
                             (0, 255, 0), 2)

                    cv2.line(frame,
                             (int(x_right_border), 0),
                             (int(x_right_border), frame_height),
                             (0, 255, 0), 2)

                    LOGGER.info(f"BORDER: {x_left_border, x_right_border}")
                    if cX < x_left_border - 20:
                        LOGGER.info("GO LEFT")
                        cv2.putText(
                                frame,
                                "GO LEFT",
                                (topLeft[0], topLeft[1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                                )

                    if x_right_border + 20 < cX:
                        LOGGER.info("GO RIGHT")
                        cv2.putText(
                                frame,
                                "GO RIGHT",
                                (topLeft[0] - 20, topLeft[1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                                )

                elif dist < 50:
                    cv2.putText(
                            frame,
                            "GO STRAIGHT",
                            (topLeft[0], topLeft[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                            )

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            LOGGER.info("Exit key pressed, exiting")
            break

    vid.release()
    cv2.destroyAllWindows()

    LOGGER.info("video release and window destroyed. Bye.")
