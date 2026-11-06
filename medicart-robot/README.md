# MediCart Robotics

This is a separate project assembled from the provided ZIP. The ZIP supplies
the Python/OpenCV vision application and Flask web dashboard. The Arduino
Mega 2560 firmware and PlatformIO configuration are included from the existing
`mega-blink` project so this project can be built and uploaded independently.

## Project layout

- `src/main.cpp` — Arduino Mega motor, 5-IR line-following, serial watchdog,
  and QR output firmware.
- `platformio.ini` — PlatformIO Mega 2560 target.
- `vision/medicart_vision.py` — camera, person detection, QR/ArUco scanning,
  and serial controller.
- `vision/web_dashboard.py` — Flask dashboard API and video stream.
- `vision/templates/dashboard.html` and `vision/static/` — dashboard UI.

## Run and upload

1. Check the wiring in `src/main.cpp`. For the first test, put the robot on a
   stand and keep a physical way to disconnect motor power.
2. From this project directory, build and upload the Mega:

   ```sh
   pio run -e megaatmega2560
   pio run -e megaatmega2560 -t upload
   ```

   If needed, add `--upload-port /dev/ttyACM0`. Your Linux user needs access to
   that serial device (typically by being in the `dialout` group and starting
   a new login session).
3. Set up Python from `vision/`:

   ```sh
   cd vision
   python3 -m venv .venv
   . .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   PyTorch installation varies by CPU/GPU platform. If the default installation
   does not fit your system, install the matching PyTorch build first using
   the official PyTorch installation selector.
4. Put `yolov8n-pose.pt` in `vision/`. The supplied ZIP does not contain model
   weights. On this machine, the existing copy can be copied with:

   ```sh
   cp ../mega-blink/vision/yolov8n-pose.pt .
   ```

   Or download the Ultralytics pose model by running:

   ```sh
   python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
   ```
5. Add a calibrated `vision/camera_calibration.json` with a measured
   `focal_length_pixels` value. No calibration file was supplied, so no value
   is guessed here. This project sets the vision STOP threshold to one foot,
   but that monocular estimate and threshold have not been validated as a safe
   stopping distance for your camera and robot.
6. Connect the camera and Arduino, close any serial monitor using the Arduino
   port, and start the application from `vision/`:

   ```sh
   python medicart_vision.py
   ```

   Open <http://127.0.0.1:5000> locally. Stop with Ctrl+C. The dashboard
   control API has no authentication; do not expose it to an untrusted
   network.

## Configuration and safety

Camera index, dimensions, and frame rate can be configured with
`MEDICART_CAMERA_INDEX`, `MEDICART_CAMERA_WIDTH`, `MEDICART_CAMERA_HEIGHT`,
and `MEDICART_CAMERA_FPS`. A video file can be selected with
`MEDICART_VIDEO_SOURCE=/path/to/video.mp4`.

The ZIP's manual-mode control path does not use human detection to determine
manual motor commands. The dashboard and serial control loop start independently
of camera and vision-model initialization, so Manual Mode remains available if
either vision component fails. Auto Mode remains stopped until camera frames and
vision inference are available; a stale or missing camera frame also stops Auto.
Serial heartbeats and keyboard commands are serviced independently of new camera
frames. Test manual
directions with wheels lifted before a ground test. Test IR polarity and motor
direction on your hardware. A camera-derived one-foot distance is not a
validated guarantee of stopping before contact; measure stopping distance at
the intended speed and provide an appropriate margin. Keep an independent
physical motor-power cutoff available. Auto safety currently evaluates people
across the full image; calibrate a forward-path region to the camera mounting and
robot width before operating near people outside the path.
