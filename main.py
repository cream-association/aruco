import cv2
import logging


logging.basicConfig()
LOGGER = logging.getLogger("aruco")
LOGGER.setLevel(logging.DEBUG)


if __name__ == "__main__":
    LOGGER.info("Starting video capture ...")
    vid = cv2.VideoCapture(0)

    while True:
        LOGGER.info("reading next frame")
        ret, frame = vid.read()

        # detecting aruco
        LOGGER.info("Loading aruco dict")
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        arucoParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

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

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            LOGGER.info("Exit key pressed, exiting")
            break

    vid.release()
    cv2.destroyAllWindows()

    LOGGER.info("video release and window destroyed. Bye.")
