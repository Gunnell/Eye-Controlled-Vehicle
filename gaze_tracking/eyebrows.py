import numpy as np
import cv2



class EyeBrow:
    """
    This class isolates the eyebrow region and initiates the pupil detection.
    """

    LEFT_BROW_POINTS = [17, 18, 19, 20, 21]
    RIGHT_BROW_POINTS = [22, 23, 24, 25, 26]

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None

        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        """Returns the midpoint between two points.

        Arguments:
            p1 (dlib.point): First point
            p2 (dlib.point): Second point
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """Isolates the eyebrow region from the rest of the face.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of the eyebrow
        """
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points]).astype(np.int32)
        self.landmark_points = region

        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        brow = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = brow[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _analyze(self, original_frame, landmarks, side, calibration):
        """Detects and isolates the eyebrow region and initializes the Pupil object.

        Arguments:
            original_frame (numpy.ndarray): Frame passed by the user
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            side (int): Indicates whether it's the left eyebrow (0) or the right eyebrow (1)
            calibration (Calibration): Manages the binarization threshold value
        """
        if side == 0:
            points = self.LEFT_BROW_POINTS
        elif side == 1:
            points = self.RIGHT_BROW_POINTS
        else:
            return

        self._isolate(original_frame, landmarks, points)

        # if not calibration.is_complete():
        #     calibration.evaluate(self.frame, side)

