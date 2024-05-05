import numpy
import cv2
import pickle
import glob
import logging

logging.basicConfig()
LOGGER = logging.getLogger("calibration")
LOGGER.setLevel(logging.DEBUG)

LOGGER.info("Welcome to camera calibration.")

obp = []
imp = []
cRow = 9
cCol = 6

objp = numpy.zeros((cRow * cCol, 3), numpy.float32)
objp[:, :2] = numpy.mgrid[0:cRow, 0:cCol].T.reshape(-1, 2)

images = glob.glob("resources/calibration_images/*.jpeg")
imageSize = None

LOGGER.info(
    f"Going to calibrate for a board of: {cRow} row / {cCol} col, with images (jpeg) in '/resources/calibration_images/'"
)

for iname in images:
    LOGGER.info(f"Start processing {iname}")
    img = cv2.imread(iname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    board, corners = cv2.findChessboardCorners(gray, (cRow, cCol), None)

    obp.append(objp)

    corners_acc = cv2.cornerSubPix(
        image=gray,
        corners=corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    imp.append(corners_acc)
    if not imageSize:
        imageSize = gray.shape[::-1]

    img = cv2.drawChessboardCorners(img, (cRow, cCol), corners_acc, board)
    cv2.imshow("Chessboard", img)
    cv2.waitKey(0)

LOGGER.info("All images have been processed, destroying cv2 window")
cv2.destroyAllWindows()

LOGGER.info("Calibrating camera")
calibration, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints=obp,
    imagePoints=imp,
    imageSize=imageSize,
    cameraMatrix=None,
    distCoeffs=None,
)

LOGGER.info("Picklet dump data to 'calibration.pckl'")
f = open("calibration.pckl", "wb")
pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), f)
f.close()

LOGGER.info("Everything done. See you.")
