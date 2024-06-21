import numpy as np
import cv2

class Pupil:
    """
    This class detects the iris of an eye and estimates
    the position of the pupil.
    """

    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        """Processes the eye frame to isolate the iris.

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye.
            threshold (int): Threshold value for binarizing the eye frame.

        Returns:
            numpy.ndarray: Processed frame with the isolated iris.
        """
        kernel = np.ones((3, 3), np.uint8)
        processed_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        processed_frame = cv2.erode(processed_frame, kernel, iterations=3)
        _, processed_frame = cv2.threshold(processed_frame, threshold, 255, cv2.THRESH_BINARY)

        return processed_frame

    def detect_iris(self, eye_frame):
        """Detects the iris and estimates its position by calculating the centroid.

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye.
        """
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        try:
            moments = cv2.moments(contours[-2])
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
        except (IndexError, ZeroDivisionError):
            self.x = None
            self.y = None
