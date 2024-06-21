import os
import threading
import time
from collections import Counter

import cv2
import numpy as np
import serial

from gaze_tracking import GazeTracking

# Initialize GazeTracking and webcam
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
new_frame = np.zeros((500, 500, 3), np.uint8)

# Initialize serial communication
try:
    ser = serial.Serial(port="/dev/cu.HC-06", baudrate=9600, timeout=1)
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    ser = None

eye_direction = ""

def track_eye_direction():
    global eye_direction
    while True:
        if ser:
            if eye_direction == "Up":
                new_frame[:] = (255, 255, 0)
                ser.write(b'F')
            elif eye_direction == "Down":
                new_frame[:] = (0, 255, 255)
                ser.write(b'B')
            elif eye_direction == "Right":
                new_frame[:] = (0, 0, 255)
                ser.write(b'L')
            elif eye_direction == "Left":
                new_frame[:] = (255, 0, 0)
                ser.write(b'R')
            elif eye_direction == "Center":
                new_frame[:] = (255, 255, 255)
                ser.write(b'F')
            time.sleep(1.5)
        print(eye_direction)
        eye_direction = ""
        time.sleep(0.5)

def determine_eye_direction(points, eye, pupil_coords, pupils_located, calibration, landmarks_ids, frame):
    global eye_direction
    point_distances = {point: np.linalg.norm(
        np.array([eye.landmarks.part(point).x, eye.landmarks.part(point).y]) - np.array(pupil_coords)) for point in points}
    if abs(eye.landmark_bottom[1] - eye.landmark_top[1]) / calibration.get_avg_height() <= 0.6 and pupils_located:
        eye_direction = "Down"
    else:
        right_group, left_group, up_group, down_group, center_group = landmarks_ids
        look_values = [
            np.mean([point_distances[x] for x in left_group]) * 0.9,
            np.mean([point_distances[x] for x in right_group]) * 0.9,
            np.mean([point_distances[x] for x in up_group]),
            np.mean([point_distances[x] for x in center_group]) * 0.72
        ]
        directions = ['Left', 'Right', 'Up', 'Center']
        eye_direction = directions[np.argmin(look_values)]
        for point in landmarks_ids[np.argmin(look_values)]:
            cv2.circle(frame, (eye.landmarks.part(point).x, eye.landmarks.part(point).y), 2, (0, 255, 0), cv2.FILLED)
    return eye_direction

left_history = []
right_history = []
text = ""

def process_webcam_frame():
    global left_history, right_history, text
    _, frame = webcam.read()
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    left_pupil, right_pupil = gaze.pupil_left_coords(), gaze.pupil_right_coords()

    if gaze.pupils_located:
        highlight_eye_landmarks(frame, gaze)
        left_dir = determine_eye_direction(
            [36, 37, 38, 39, 40, 41], gaze.eye_left, left_pupil, gaze.pupils_located, gaze.calibration,
            [[36, 37, 41], [38, 40, 39], [37, 38], [40, 41], [41, 40]], frame)
        right_dir = determine_eye_direction(
            [42, 43, 44, 45, 46, 47], gaze.eye_right, right_pupil, gaze.pupils_located, gaze.calibration,
            [[42, 43, 47], [46, 44, 45], [43, 44], [46, 47], [46, 47]], frame)
        update_eye_direction_history(left_dir, right_dir)

        if not gaze.calibration.is_complete():
            cv2.putText(frame, "Calibrating...", (90, 100), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        draw_face_rectangle(frame, gaze)

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    cv2.imshow("Direction", frame)

def highlight_eye_landmarks(frame, gaze):
    left_eye, right_eye = gaze.eye_left, gaze.eye_right
    for eye in [left_eye, right_eye]:
        for point in eye.landmark_points:
            cv2.circle(frame, point, 2, (0, 0, 255), cv2.FILLED)
    for brow in [gaze.brow_left, gaze.brow_right]:
        for point in brow.landmark_points:
            cv2.circle(frame, point, 2, (0, 0, 255), cv2.FILLED)

def update_eye_direction_history(left_dir, right_dir):
    global left_history, right_history, text
    left_history.append(left_dir)
    right_history.append(right_dir)
    if len(left_history) % 10 == 0 and left_history:
        left_most_dir = Counter(left_history).most_common(1)[0][0]
        right_most_dir = Counter(right_history).most_common(1)[0][0]
        left_history, right_history = [], []
        text = left_most_dir if left_most_dir == right_most_dir else "Unknown"

def draw_face_rectangle(frame, gaze):
    try:
        face = gaze.face
        x, y, w, h = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    except Exception:
        pass

def main():
    eye_thread = threading.Thread(target=track_eye_direction)
    eye_thread.start()
    while True:
        key = cv2.waitKey(1)
        process_webcam_frame()
        if key == 27:  # ESC key to exit
            break
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
