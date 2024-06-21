from __future__ import division
import cv2
import numpy as np
from .pupil import Pupil


class Calibration:
    """
    This class calibrates the pupil detection algorithm by finding the
    best binarization threshold value for the person and the webcam.
    """

    def __init__(self):
        self.nb_frames = 20  # Number of frames used for calibration
        self.thresholds_left = []  # List to store threshold values for the left eye
        self.thresholds_right = []  # List to store threshold values for the right eye
        self.eye_heights = []  # List to store the heights of the eyes

    def is_complete(self) -> bool:
        """Check if calibration is completed by ensuring enough frames have been processed"""
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self, side: int) -> int:
        """Calculate and return the average threshold value for the specified eye.

        Argument:
            side (int): 0 for left eye, 1 for right eye
        """
        if side == 0:
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1:
            return int(sum(self.thresholds_right) / len(self.thresholds_right))
        else:
            raise ValueError("Invalid value for side. Use 0 for left eye and 1 for right eye.")

    def get_avg_height(self) -> float:
        """Calculate and return the average height of the eyes during calibration"""
        return np.mean(self.eye_heights)

    @staticmethod
    def iris_size(frame: np.ndarray) -> float:
        """Calculate the percentage of the eye's surface occupied by the iris.

        Argument:
            frame (numpy.ndarray): Binarized iris frame
        """
        frame = frame[5:-5, 5:-5]  # Crop the frame to exclude the borders
        height, width = frame.shape[:2]
        num_pixels = height * width  # Total number of pixels in the cropped frame
        num_blacks = num_pixels - cv2.countNonZero(frame)  # Number of black pixels
        return num_blacks / num_pixels  # Ratio of black pixels to total pixels

    @staticmethod
    def find_best_threshold(eye_frame: np.ndarray) -> int:
        """Determine the optimal threshold to binarize the eye frame.

        Argument:
            eye_frame (numpy.ndarray): Frame of the eye to be analyzed
        """
        average_iris_size = 0.48  # Target iris size ratio
        trials = {}  # Dictionary to store iris size for each threshold

        for threshold in range(5, 100, 5):  # Test thresholds from 5 to 95
            try:
                iris_frame = Pupil.image_processing(eye_frame, threshold)
                trials[threshold] = Calibration.iris_size(iris_frame)
            except Exception as e:
                print(f"Error processing threshold {threshold}: {e}")
                continue

        # Select the threshold with the iris size closest to the target
        best_threshold, _ = min(trials.items(), key=lambda p: abs(p[1] - average_iris_size))
        return best_threshold

    def evaluate(self, eye_frame: np.ndarray, side: int, height: float):
        """Improve calibration by evaluating a given eye frame.

        Arguments:
            eye_frame (numpy.ndarray): Frame of the eye
            side (int): 0 for left eye, 1 for right eye
            height (float): Height of the eye
        """
        threshold = self.find_best_threshold(eye_frame)  # Find the best threshold for the eye frame

        if side == 0:
            self.thresholds_left.append(threshold)  # Add threshold to the left eye list
        elif side == 1:
            self.thresholds_right.append(threshold)  # Add threshold to the right eye list
        else:
            raise ValueError("Invalid value for side. Use 0 for left eye and 1 for right eye.")

        self.eye_heights.append(height)  # Add the eye height to the list
