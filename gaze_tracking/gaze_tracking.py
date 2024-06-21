from __future__ import division
import cv2
import dlib
from .eye import Eye
from .calibration import Calibration
from .eyebrows import EyeBrow

class GazeTracking:
    """
    This class tracks the user's gaze, providing information about the position of the eyes
    and pupils, and whether the eyes are open or closed.
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.brow_left = None
        self.brow_right = None
        self.calibration = Calibration()

        # Initialize the face detector and facial landmarks predictor
        self._face_detector = dlib.get_frontal_face_detector()
        model_path = "shape_predictor_68_face_landmarks.dat"
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """Checks if pupils have been located."""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initializes Eye objects."""
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(gray_frame)

        try:
            self.face = faces[0]
            landmarks = self._predictor(gray_frame, faces[0])
            self.eye_left = Eye(gray_frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(gray_frame, landmarks, 1, self.calibration)
            self.brow_left = EyeBrow(gray_frame, landmarks, 0, self.calibration)
            self.brow_right = EyeBrow(gray_frame, landmarks, 1, self.calibration)
        except IndexError:
            self.eye_left = None
            self.eye_right = None
            self.brow_left = None
            self.brow_right = None

    def refresh(self, frame):
        """Updates the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze.
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil."""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil."""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 indicating horizontal gaze direction."""
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 indicating vertical gaze direction."""
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns True if the user is looking to the right."""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns True if the user is looking to the left."""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_top(self):
        """Returns True if the user is looking to the top."""
        if self.pupils_located:
            return self.vertical_ratio() <= 0.35

    def is_bottom(self):
        """Returns True if the user is looking to the bottom."""
        if self.pupils_located:
            return self.vertical_ratio() >= 0.65

    def is_center(self):
        """Returns True if the user is looking to the center."""
        if self.pupils_located:
            return not self.is_right() and not self.is_left()

    def is_blinking(self):
        """Returns True if the user is blinking."""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Returns the frame with highlighted pupils."""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame
