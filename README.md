```markdown
# Eye Tracking with Gaze Tracking for Arduino Vehicle Control

This project utilizes computer vision and gaze tracking to detect the direction of a user's eye movements and control an Arduino-powered vehicle. The purpose of this project is to enable vehicle movement based on the user's eye direction, providing an innovative hands-free control method. The project is built using Python and leverages libraries such as OpenCV, Dlib, and the `gaze_tracking` package.

## Features

- Tracks the user's eye movements (left, right, up, down, center).
- Controls an Arduino vehicle through serial communication based on the detected eye direction.
- Displays the direction of the gaze in real-time.
- Highlights eye and eyebrow landmarks on the video feed.

## Requirements

- Python 3.x
- OpenCV
- Dlib
- NumPy
- PySerial
- Arduino (with HC-06 Bluetooth module)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Gunnell/Eye-Controlled-Vehicle.git
   cd eye-tracking-arduino
   ```

2. **Install the required packages:**

   ```bash
   pip install opencv-python dlib numpy  pyserial
   ```

## Usage

1. **Set up your Arduino vehicle:**

   - Connect the HC-06 Bluetooth module to the Arduino.
   - Upload the Arduino sketch to your Arduino board to handle the serial commands for vehicle movement.

2. **Run the main script:**

   ```bash
   python main.py
   ```

3. **Interacting with the Program:**

   - The program will start the webcam and display the video feed.
   - It will detect and track your eye movements and display the direction on the screen.
   - The detected direction will be sent via Bluetooth to control the Arduino vehicle.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


```
