import cv2
import logging


logging.basicConfig()
LOGGER = logging.getLogger("aruco")
LOGGER.setLevel(logging.DEBUG)


if __name__ == "__main__":
    LOGGER.info("Starting video capture ...")
    vid = cv2.VideoCapture(0)

    while True:
        LOGGER.info("While loop, start reading video")
        ret, frame = vid.read()

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            LOGGER.info("Exit key pressed, exiting")
            break

    vid.release()
    cv2.destroyAllWindows()

    LOGGER.info("video release and window destroyed. Bye.")
