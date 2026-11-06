import json
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from importlib import import_module
from math import hypot
from pathlib import Path
from statistics import median
from typing import Optional

import cv2
import numpy as np
import serial
from web_dashboard import WebDashboard

SERIAL_PORT = "/dev/ttyACM0"
SERIAL_PORT_CANDIDATES = ("/dev/ttyACM0", "/dev/ttyACM1")
SERIAL_BAUD = 115200
CAMERA_INDEX = int(os.environ.get("MEDICART_CAMERA_INDEX", "0"))
# NOTE: focal_length_pixels in camera_calibration.json is only valid at the resolution it was
# calibrated at. Keep 1280x720 unless you recalibrate (the web stream is downscaled anyway).
FRAME_WIDTH = int(os.environ.get("MEDICART_CAMERA_WIDTH", "1280"))
FRAME_HEIGHT = int(os.environ.get("MEDICART_CAMERA_HEIGHT", "720"))
CAMERA_TARGET_FPS = int(os.environ.get("MEDICART_CAMERA_FPS", "30"))
PROJECT_DIRECTORY = Path(__file__).resolve().parent
MOTOR_DIRECTORY = PROJECT_DIRECTORY.parent / "motor"
PROJECT_ROOT = PROJECT_DIRECTORY.parent
ROBOTICS_DIRECTORY = PROJECT_DIRECTORY.parent.parent
FALLBACK_MODEL_PATH = next(
    (path for path in (PROJECT_DIRECTORY / "yolov8n.pt", PROJECT_ROOT / "yolov8n.pt",
                       MOTOR_DIRECTORY / "yolov8n.pt",
                       ROBOTICS_DIRECTORY / "yolov8n.pt") if path.is_file()),
    PROJECT_DIRECTORY / "yolov8n.pt",
)
POSE_MODEL_PATH = next(
    (path for path in (PROJECT_DIRECTORY / "yolov8n-pose.pt", PROJECT_ROOT / "yolov8n-pose.pt",
                       MOTOR_DIRECTORY / "yolov8n-pose.pt",
                       ROBOTICS_DIRECTORY / "yolov8n-pose.pt") if path.is_file()),
    PROJECT_DIRECTORY / "yolov8n-pose.pt",
)
CALIBRATION_FILE = next(
    (path for path in (PROJECT_DIRECTORY / "camera_calibration.json", PROJECT_ROOT / "camera_calibration.json",
                       MOTOR_DIRECTORY / "camera_calibration.json",
                       ROBOTICS_DIRECTORY / "camera_calibration.json") if path.is_file()),
    PROJECT_DIRECTORY / "camera_calibration.json",
)
INFERENCE_SIZE = 256
CPU_THREADS = 2
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.35
REAL_PERSON_HEIGHT = 1.76784
FEET_PER_METER = 3.28084
STOP_DISTANCE = 1.0 / FEET_PER_METER     # 1 foot = 0.3048 m  (raise this for more braking margin)
WARNING_DISTANCE = 1.0                   # display only; does not change robot behaviour
HEARTBEAT_INTERVAL_SECONDS = 0.1         # Arduino times out at 750 ms
SERIAL_RETRY_INTERVAL_SECONDS = 2.0
SERIAL_BOOT_WAIT_SECONDS = 2.5
TERMINAL_REPORT_INTERVAL_SECONDS = float(os.environ.get("MEDICART_MONITOR_INTERVAL", "1.0"))
QR_SCAN_INTERVAL = 5          # every Nth NEW camera frame
QR_RETRY_INTERVAL = 3
QR_LOST_SCAN_LIMIT = 3
QR_CONFIRMATION_WINDOW = 3
QR_CONFIRMATION_REQUIRED_SCANS = 2

CLOSE_BBOX_FRACTION = 0.60
SAFETY_HOLD_SECONDS = 1.0
INFERENCE_STALE_SECONDS = 1.0
STARTUP_GRACE_SECONDS = 3.0

VIDEO_WINDOW_NAME = "MediCart Vision"
KEYPOINT_NAMES = (
    "Nose", "L-eye", "R-eye", "L-ear", "R-ear", "L-shoulder", "R-shoulder",
    "L-elbow", "R-elbow", "L-wrist", "R-wrist", "L-hip", "R-hip", "L-knee",
    "R-knee", "L-ankle", "R-ankle",
)
SKELETON_LINKS = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)
PERSON_COLORS = ((255, 180, 50), (60, 220, 120), (220, 120, 220), (80, 200, 240), (220, 180, 70))


@dataclass
class PersonDetection:
    person_id: str
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    width: int
    height: int
    confidence: float
    distance: Optional[float]
    horizontal_position: str
    vertical_position: str
    partial: bool = False
    distance_from_partial_pose: bool = False
    keypoints: tuple[tuple[float, float, float], ...] = ()
    close: bool = False


def is_display_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def initialize_camera() -> cv2.VideoCapture:
    source = (
        os.environ.get("MOTOR_VIDEO_SOURCE")
        or os.environ.get("MEDICART_VIDEO_SOURCE")
        or os.environ.get("VIDEO_SOURCE")
        or (sys.argv[1] if len(sys.argv) > 1 else None)
    )
    if source:
        if str(source).isdigit():
            camera = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
            if camera.isOpened():
                return configure_camera(camera)
            camera.release()
        source_path = Path(str(source)).expanduser()
        if source_path.is_file():
            camera = cv2.VideoCapture(str(source_path))
            if camera.isOpened():
                return configure_camera(camera)
            camera.release()
        raise RuntimeError(f"Could not open requested camera/video source: {source}")

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}; check connection or camera permissions.")
    return configure_camera(camera)


def configure_camera(camera: cv2.VideoCapture) -> cv2.VideoCapture:
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_TARGET_FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


class LatestFrameCapture:
    """Continuously drain the camera and retain only its newest frame."""

    def __init__(self, camera: cv2.VideoCapture) -> None:
        self.camera = camera
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame: Optional[np.ndarray] = None
        self._captured_at = 0.0
        self._fps = 0.0
        self._error: Optional[str] = None
        self._thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _capture_loop(self) -> None:
        previous_capture = 0.0
        while not self._stop_event.is_set():
            ok, frame = self.camera.read()
            captured_at = time.monotonic()
            if not ok or frame is None:
                with self._lock:
                    self._error = "camera returned no frame"
                self._stop_event.wait(0.05)
                continue
            with self._lock:
                if previous_capture:
                    interval = captured_at - previous_capture
                    measured_fps = 1.0 / interval if interval > 0 else 0.0
                    self._fps = measured_fps if self._fps == 0 else 0.8 * self._fps + 0.2 * measured_fps
                self._frame = frame
                self._captured_at = captured_at
                self._error = None
            previous_capture = captured_at

    def latest(self) -> tuple[Optional[np.ndarray], float, float, Optional[str]]:
        with self._lock:
            return self._frame, self._captured_at, self._fps, self._error

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self.camera.release()
        self._thread.join(timeout=1.0)


def motor_label(speed: int | str) -> str:
    if isinstance(speed, str):
        return speed
    if speed > 0:
        return f"FWD {speed:03d}"
    if speed < 0:
        return f"REV {-speed:03d}"
    return "STOP"


def log_monitor_row(elapsed: float, mode: str, keys: str, left_speed: int | str, right_speed: int | str,
                    camera_fps: float, frame_age_ms: float, inference_ms: Optional[float], status: str) -> None:
    print(
        f"{elapsed:6.1f} | {mode:6.6} | {keys:12.12} | {motor_label(left_speed):8.8} | "
        f"{motor_label(right_speed):8.8} | {camera_fps:6.1f} | {frame_age_ms:8.1f} | "
        f"{inference_ms if inference_ms is not None else 0.0:8.1f} | {status:25.25}",
        flush=True,
    )


def log_monitor_header() -> None:
    print(
        f"{'TIME(s)':>6} | {'CONTROL':6} | {'KEY':12} | {'MOTOR_L':8} | {'MOTOR_R':8} | "
        f"{'CAM_FPS':>7} | {'FRAME_MS':>8} | {'INFER_MS':>8} | {'STATUS':25}",
        flush=True,
    )


# ----------------------------------------------------------------------------
# Serial helpers
# ----------------------------------------------------------------------------

def find_serial_port() -> Optional[str]:
    for candidate in SERIAL_PORT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def connect_serial(baud: int = SERIAL_BAUD):
    port = find_serial_port()
    if port is None:
        print(f"[WARN] No Arduino port found (looked for {', '.join(SERIAL_PORT_CANDIDATES)})", flush=True)
        return None
    if port != SERIAL_PORT:
        print(f"[SERIAL] Arduino is on {port}, not {SERIAL_PORT}", flush=True)
    try:
        try:
            arduino = serial.Serial(port, baud, timeout=0.2, write_timeout=0.1, exclusive=True)
        except TypeError:
            print("[WARN] pyserial too old for exclusive=True; opening WITHOUT a port lock", flush=True)
            arduino = serial.Serial(port, baud, timeout=0.2, write_timeout=0.1)
        print(f"[INFO] Connected to {port} at {baud} baud", flush=True)
        return arduino
    except (serial.SerialException, OSError) as exc:
        print(f"[WARN] Could not open {port}: {exc}", flush=True)
        print("[HINT] Another program probably owns the port. Run:", flush=True)
        print(f"         fuser -k {port}", flush=True)
        print("         sudo systemctl stop ModemManager", flush=True)
        return None


def close_serial(arduino: Optional[serial.Serial]) -> None:
    if arduino is None:
        return
    try:
        arduino.close()
    except (serial.SerialException, OSError):
        pass


def send_command(arduino: Optional[serial.Serial], command: str) -> bool:
    if arduino is None or not arduino.is_open:
        return False
    try:
        # No flush(): it blocks until the OS buffer drains and just adds latency to the main loop.
        arduino.write((command + "\n").encode("utf-8"))
        return True
    except (serial.SerialException, OSError) as exc:
        print(f"[WARN] Serial write failed for {command}: {exc}", flush=True)
        return False


def read_arduino_lines(arduino: serial.Serial, rx_buffer: bytearray) -> Optional[list[str]]:
    try:
        waiting = arduino.in_waiting
        if waiting:
            rx_buffer.extend(arduino.read(waiting))
    except (serial.SerialException, OSError) as exc:
        print(f"[WARN] Serial read failed: {exc}", flush=True)
        return None

    lines = []
    while b"\n" in rx_buffer:
        newline_index = rx_buffer.index(b"\n")
        raw = bytes(rx_buffer[:newline_index])
        del rx_buffer[:newline_index + 1]
        text = raw.decode("utf-8", errors="ignore").strip()
        if text:
            lines.append(text)
    if len(rx_buffer) > 512:
        rx_buffer.clear()
    return lines


def load_model():
    try:
        torch = import_module("torch")
        YOLO = import_module("ultralytics").YOLO
    except ImportError as exc:
        raise RuntimeError(
            "YOLO dependencies are missing. Activate vision/.venv and install "
            "the packages in vision/requirements.txt."
        ) from exc

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        torch.set_num_threads(CPU_THREADS)
    if POSE_MODEL_PATH.is_file():
        model_path = POSE_MODEL_PATH
    elif FALLBACK_MODEL_PATH.is_file():
        model_path = FALLBACK_MODEL_PATH
        print("[WARN] Pose weights missing; fallback detections lack pose distances and trigger STOP.", flush=True)
    else:
        raise RuntimeError(f"YOLO weights not found: {POSE_MODEL_PATH}")
    print(f"[INFO] Loading YOLO model: {model_path}", flush=True)
    return YOLO(str(model_path)), (0 if use_cuda else "cpu")


def load_calibration() -> Optional[float]:
    try:
        with CALIBRATION_FILE.open("r", encoding="utf-8") as file:
            calibration = json.load(file)
        focal_length = float(calibration["focal_length_pixels"])
        if focal_length <= 0:
            raise ValueError("focal length must be positive")
        return focal_length
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"[WARN] Invalid or missing camera calibration {CALIBRATION_FILE}: {exc}", flush=True)
        return None


def calculate_distance(pixel_height: Optional[float], real_height: float = REAL_PERSON_HEIGHT,
                       focal_length: Optional[float] = None) -> Optional[float]:
    if pixel_height is None or pixel_height <= 0 or focal_length is None or focal_length <= 0:
        return None
    return (real_height * focal_length) / pixel_height


def format_distance(distance: float) -> str:
    return f"{distance:.2f} m / {distance * FEET_PER_METER:.2f} ft"


def estimate_full_person_pixel_height(keypoints: Optional[list[list[float]]], visible_height: int) -> Optional[float]:
    if not keypoints:
        return None

    def joint(index: int) -> Optional[tuple[float, float]]:
        if index >= len(keypoints) or len(keypoints[index]) < 3:
            return None
        x, y, confidence = keypoints[index][:3]
        return (float(x), float(y)) if float(confidence) >= 0.35 else None

    def midpoint(indices: tuple[int, int]) -> Optional[tuple[float, float]]:
        points = [point for index in indices if (point := joint(index)) is not None]
        if not points:
            return None
        return sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)

    head = joint(0)
    shoulders = midpoint((5, 6))
    hips = midpoint((11, 12))
    knees = midpoint((13, 14))
    ankles = midpoint((15, 16))
    segments = (
        (head, shoulders, 0.12), (shoulders, hips, 0.29), (hips, knees, 0.245),
        (knees, ankles, 0.246), (hips, ankles, 0.49), (shoulders, knees, 0.535),
        (head, ankles, 0.92),
    )
    estimates = [
        hypot(end[0] - start[0], end[1] - start[1]) / fraction
        for start, end, fraction in segments
        if start is not None and end is not None
        and hypot(end[0] - start[0], end[1] - start[1]) / fraction >= visible_height * 0.95
    ]
    return median(estimates) if estimates else None


def calculate_position(center: tuple[int, int], frame_width: int, frame_height: int) -> tuple[str, str]:
    x, y = center
    horizontal = "LEFT" if x < frame_width / 3 else "RIGHT" if x > frame_width * 2 / 3 else "CENTER"
    vertical = "TOP" if y < frame_height / 3 else "BOTTOM" if y > frame_height * 2 / 3 else "CENTER"
    return horizontal, vertical


def detect_people(model, frame, focal_length: Optional[float], device,
                  track_labels: dict[int, str], next_label: int) -> tuple[list[PersonDetection], int]:
    results = model.track(
        frame, persist=True, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD,
        imgsz=INFERENCE_SIZE, device=device, max_det=20, verbose=False,
    )
    if not results or results[0].boxes is None:
        return [], next_label

    frame_height, frame_width = frame.shape[:2]
    keypoint_data = getattr(results[0], "keypoints", None)
    detections = []
    for box_index, box in enumerate(results[0].boxes):
        confidence = float(box.conf[0].item())
        if confidence < CONFIDENCE_THRESHOLD or int(box.cls[0].item()) != PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))
        visible_height = y2 - y1
        if x2 <= x1 or visible_height <= 0:
            continue

        pose = None
        if keypoint_data is not None and keypoint_data.data is not None and box_index < len(keypoint_data.data):
            pose = keypoint_data.data[box_index].detach().cpu().tolist()
        pose_partial = False
        if pose and len(pose) >= 17:
            head_visible = pose[0][2] >= 0.35
            ankle_visible = pose[15][2] >= 0.35 or pose[16][2] >= 0.35
            pose_partial = not head_visible or not ankle_visible
        touches_edge = x1 <= 1 or y1 <= 1 or x2 >= frame_width - 2 or y2 >= frame_height - 2
        partial = touches_edge or pose_partial

        close_to_camera = (
            visible_height > CLOSE_BBOX_FRACTION * frame_height
            or (x2 - x1) > CLOSE_BBOX_FRACTION * frame_width
        )

        pose_height = estimate_full_person_pixel_height(pose, visible_height) if pose else None
        person_pixel_height = pose_height if pose_height is not None else visible_height
        distance = calculate_distance(person_pixel_height, focal_length=focal_length) if not partial or pose_height is not None else None
        distance_from_partial_pose = partial and pose_height is not None

        tracker_id = int(box.id[0].item()) if box.id is not None else None
        if tracker_id is None:
            person_id = f"Person {next_label}"
            next_label += 1
        else:
            if tracker_id not in track_labels:
                track_labels[tracker_id] = f"Person {next_label}"
                next_label += 1
            person_id = track_labels[tracker_id]

        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        horizontal, vertical = calculate_position(center, frame_width, frame_height)
        points = tuple(
            (float(point[0]), float(point[1]), float(point[2]))
            for point in pose or ()
        )
        detections.append(PersonDetection(
            person_id=person_id,
            bbox=(x1, y1, x2, y2),
            center=center,
            width=x2 - x1,
            height=visible_height,
            confidence=confidence,
            distance=distance,
            horizontal_position=horizontal,
            vertical_position=vertical,
            partial=partial,
            distance_from_partial_pose=distance_from_partial_pose,
            keypoints=points,
            close=close_to_camera,
        ))
    return detections, next_label


def find_nearest_person(people: list[PersonDetection]) -> Optional[PersonDetection]:
    valid = [person for person in people if person.distance is not None]
    return min(valid, key=lambda person: person.distance or float("inf")) if valid else None


def get_robot_status(people: list[PersonDetection], nearest: Optional[PersonDetection] = None) -> str:
    if not people:
        return "SAFE"
    if any(person.close for person in people):
        return "STOP"
    nearest = nearest or find_nearest_person(people)
    nearest_distance = nearest.distance if nearest is not None else None
    if nearest_distance is not None and nearest_distance <= STOP_DISTANCE:
        return "STOP"
    if any(person.distance is None for person in people) or nearest_distance is None:
        return "PERSON_DETECTED"
    if nearest_distance <= WARNING_DISTANCE:
        return "WARNING"
    return "SAFE"


def arduino_safety_command(status: str) -> str:
    if status in ("STOP", "PERSON_DETECTED"):
        return "STOP"
    return "SAFE"


def heartbeat_due(now: float, last_heartbeat: float) -> bool:
    return now - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS


def qr_command(value: int) -> Optional[str]:
    return f"QR:{value}" if 1 <= value <= 10 else None


def pending_qr_command(qr_value: Optional[str], last_sent_value: Optional[str]) -> Optional[str]:
    if qr_value is None or qr_value == last_sent_value:
        return None
    return qr_command(int(qr_value))


def initialize_qr_detectors():
    aruco = cv2.aruco
    parameters = aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.02
    parameters.adaptiveThreshWinSizeMax = 33
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    marker_detector = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(aruco.DICT_4X4_50), parameters
    )
    return cv2.QRCodeDetector(), marker_detector, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def scan_qr_and_markers(frame, frame_count, qr_detector, marker_detector, contrast_enhancer):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = marker_detector.detectMarkers(gray)
    qr_data = None
    detection_type = "ARUCO MARKER"
    boundary = None

    if ids is not None:
        for index, marker_id in enumerate(ids.flatten()):
            if 1 <= int(marker_id) <= 10:
                qr_data = str(int(marker_id))
                boundary = corners[index].reshape(-1, 2)
                break

    if qr_data is None and frame_count % QR_RETRY_INTERVAL == 0:
        enhanced = contrast_enhancer.apply(gray)
        enlarged = cv2.resize(enhanced, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        retry_corners, retry_ids, _ = marker_detector.detectMarkers(enlarged)
        if retry_ids is not None:
            for index, marker_id in enumerate(retry_ids.flatten()):
                if 1 <= int(marker_id) <= 10:
                    qr_data = str(int(marker_id))
                    boundary = retry_corners[index].reshape(-1, 2) / 1.5
                    break

    if qr_data is None:
        qr_text, qr_points, _ = qr_detector.detectAndDecode(gray)
        if not qr_text:
            if qr_points is not None:
                points = qr_points.reshape(-1, 2)
                x_min, y_min = points.min(axis=0).astype(int)
                x_max, y_max = points.max(axis=0).astype(int)
                margin = max(4, int(max(x_max - x_min, y_max - y_min) * 0.12))
                crop_x1, crop_y1 = max(0, x_min - margin), max(0, y_min - margin)
                crop_x2 = min(gray.shape[1], x_max + margin + 1)
                crop_y2 = min(gray.shape[0], y_max + margin + 1)
                qr_crop = gray[crop_y1:crop_y2, crop_x1:crop_x2]
                scale = max(2.0, min(4.0, 160.0 / max(qr_crop.shape)))
                retry_image = cv2.resize(qr_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                qr_text, retry_points, _ = qr_detector.detectAndDecode(retry_image)
                if retry_points is not None:
                    qr_points = retry_points / scale
                    qr_points[..., 0] += crop_x1
                    qr_points[..., 1] += crop_y1
            else:
                retry_image = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                qr_text, retry_points, _ = qr_detector.detectAndDecode(retry_image)
                if retry_points is not None:
                    qr_points = retry_points / 1.5

        if qr_points is not None:
            boundary = qr_points.reshape(-1, 2)
        if isinstance(qr_text, str) and qr_text.strip().isdigit():
            value = int(qr_text.strip())
            if 1 <= value <= 10:
                qr_data = str(value)
                detection_type = "QR CODE"

    return qr_data, detection_type, boundary if qr_data is not None else None


def update_qr_confirmation(qr_data, detection_type, recent_values, scans_without_code,
                           last_qr_type, last_qr_data):
    current_value = (detection_type, qr_data) if qr_data is not None else None
    recent_values.append(current_value)
    if qr_data is not None:
        scans_without_code = 0
        if recent_values.count(current_value) >= QR_CONFIRMATION_REQUIRED_SCANS:
            last_qr_type, last_qr_data = detection_type, qr_data
    else:
        scans_without_code += 1
        if scans_without_code >= QR_LOST_SCAN_LIMIT:
            last_qr_type = last_qr_data = None
            recent_values.clear()
    return scans_without_code, last_qr_type, last_qr_data


def draw_camera_reference(frame):
    height, width = frame.shape[:2]
    camera_point = (width // 2, height - 24)
    cv2.line(frame, camera_point, (int(width * 0.08), 0), (95, 95, 95), 1)
    cv2.line(frame, camera_point, (int(width * 0.92), 0), (95, 95, 95), 1)
    cv2.circle(frame, camera_point, 7, (255, 255, 255), -1)
    cv2.circle(frame, camera_point, 10, (25, 25, 25), 2)
    cv2.putText(frame, "CAMERA", (camera_point[0] + 13, camera_point[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return camera_point


def draw_distance_arrow(frame, camera_point, person, color):
    cv2.arrowedLine(frame, camera_point, person.center, color, 2, cv2.LINE_AA, tipLength=0.04)
    if person.distance is None:
        label = f"{person.person_id}: partial" if person.partial else f"{person.person_id}: calibrate"
    elif person.distance_from_partial_pose:
        label = f"{person.person_id}: ~{format_distance(person.distance)}"
    else:
        label = f"{person.person_id}: {format_distance(person.distance)}"
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    text_x = max(4, min(frame.shape[1] - text_size[0] - 8, person.center[0] + 10))
    text_y = max(text_size[1] + 8, person.center[1] - 12)
    cv2.rectangle(frame, (text_x - 4, text_y - text_size[1] - 6),
                  (text_x + text_size[0] + 4, text_y + 4), (20, 20, 20), -1)
    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255, 255, 255), 2, cv2.LINE_AA)


def draw_person_box(frame, person, is_nearest, color):
    x1, y1, x2, y2 = person.bbox
    box_color = (0, 128, 255) if person.partial else (0, 0, 255) if is_nearest else color
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4 if is_nearest or person.partial else 2)
    if person.distance is None:
        distance_text = "partial: distance unknown" if person.partial else "uncalibrated"
    else:
        prefix = "~" if person.distance_from_partial_pose else ""
        distance_text = f"{prefix}{format_distance(person.distance)}"
    lines = (
        f"{person.person_id}  {distance_text}",
        f"{person.horizontal_position}/{person.vertical_position}  {person.confidence:.0%}",
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 22
    label_height = len(lines) * line_height + 8
    label_width = max(cv2.getTextSize(line, font, 0.58, 2)[0][0] for line in lines) + 12
    label_x = max(0, min(x1, frame.shape[1] - label_width))
    label_y = max(label_height, y1)
    cv2.rectangle(frame, (label_x, label_y - label_height),
                  (label_x + label_width, label_y), box_color, -1)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (label_x + 6, label_y - label_height + 20 + index * line_height),
                    font, 0.58, (255, 255, 255), 2, cv2.LINE_AA)


def draw_person_keypoints(frame, person):
    if not person.keypoints:
        return
    visible = [point[2] >= 0.35 for point in person.keypoints[:len(KEYPOINT_NAMES)]]
    for start, end in SKELETON_LINKS:
        if end < len(visible) and visible[start] and visible[end]:
            point_a = tuple(int(round(value)) for value in person.keypoints[start][:2])
            point_b = tuple(int(round(value)) for value in person.keypoints[end][:2])
            cv2.line(frame, point_a, point_b, (40, 230, 230), 2, cv2.LINE_AA)
    for index, (x, y, confidence) in enumerate(person.keypoints[:len(KEYPOINT_NAMES)]):
        if confidence < 0.35:
            continue
        point = (int(round(x)), int(round(y)))
        cv2.circle(frame, point, 4, (0, 40, 255), -1, cv2.LINE_AA)
        label_x = max(2, min(point[0] + 5, frame.shape[1] - 58))
        label_y = max(12, min(point[1] - 5, frame.shape[0] - 4))
        cv2.putText(frame, KEYPOINT_NAMES[index], (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)


def draw_qr_detection(frame, boundary, detection_type, qr_data):
    if boundary is None or qr_data is None:
        return
    polygon = boundary.astype(int).reshape(-1, 1, 2)
    cv2.polylines(frame, [polygon], True, (0, 255, 0), 3, cv2.LINE_AA)
    x_min, y_min = boundary.reshape(-1, 2).min(axis=0).astype(int)
    label = f"{detection_type}: {qr_data}"
    size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    text_x = max(4, min(int(x_min), frame.shape[1] - size[0] - 12))
    text_y = max(size[1] + 10, int(y_min) - 8)
    cv2.rectangle(frame, (text_x - 4, text_y - size[1] - 6),
                  (text_x + size[0] + 4, text_y + 4), (0, 120, 0), -1)
    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)


def draw_status(frame, people, nearest, status, fps, focal_length):
    if nearest is None:
        nearest_text = "NEAREST PERSON: UNKNOWN"
        distance_text = "DISTANCE: N/A"
    else:
        nearest_text = f"NEAREST PERSON: {nearest.person_id}"
        distance_text = f"DISTANCE: {format_distance(nearest.distance)}" if nearest.distance is not None else "DISTANCE: N/A"
    focal_text = f"FOCAL: {focal_length:.1f} px" if focal_length else "FOCAL: not calibrated"
    lines = (
        nearest_text, distance_text, f"STATUS: {status}",
        f"PEOPLE: {len(people)}   FPS: {fps:.1f}", focal_text,
    )
    panel_height = len(lines) * 24 + 12
    cv2.rectangle(frame, (10, 10), (350, 10 + panel_height), (20, 20, 20), -1)
    status_color = (50, 220, 90) if status in ("SAFE", "MANUAL") else (0, 165, 255)
    if status == "STOP" or status == "PERSON_DETECTED":
        status_color = (0, 0, 255)
    for index, line in enumerate(lines):
        color = status_color if line.startswith("STATUS:") else (245, 245, 245)
        cv2.putText(frame, line, (20, 34 + index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, color, 2, cv2.LINE_AA)
    if status == "MANUAL":
        cv2.putText(frame, "MANUAL: human detection OFF", (20, 10 + panel_height + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (235, 235, 235), 2, cv2.LINE_AA)
    elif not people:
        cv2.putText(frame, "No person detected", (20, 10 + panel_height + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (235, 235, 235), 2, cv2.LINE_AA)


def main():
    arduino = None
    camera = None
    camera_capture = None
    executor = None
    camera_init_future: Optional[Future] = None
    model_init_future: Optional[Future] = None
    qr_init_future: Optional[Future] = None
    dashboard = WebDashboard()
    display_available = is_display_available()

    try:
        last_serial_attempt = time.monotonic()
        arduino = connect_serial()
        serial_ready_at = time.monotonic() + SERIAL_BOOT_WAIT_SECONDS if arduino is not None else 0.0
        had_connection = arduino is not None
        last_port_name: Optional[str] = arduino.port if arduino is not None else None
        rx_buffer = bytearray()
        boot_messages_seen = 0
        arduino_status = "WAITING"
        arduino_left_speed = 0
        arduino_right_speed = 0
        if arduino is not None:
            print(f"[SERIAL] connected, waiting {SERIAL_BOOT_WAIT_SECONDS:.1f}s for Mega boot", flush=True)

        if not dashboard.start():
            raise RuntimeError("Could not start the dashboard server on 127.0.0.1:5000.")

        executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="medicart-vision")
        camera_init_future = executor.submit(initialize_camera)
        model_init_future = executor.submit(load_model)
        qr_init_future = executor.submit(initialize_qr_detectors)
        model = None
        model_error: Optional[str] = None
        device = "cpu"
        qr_detector = None
        marker_detector = None
        contrast_enhancer = None
        focal_length = load_calibration()
        if focal_length is None:
            print("[WARN] Distance calibration unavailable; detected people will trigger STOP.", flush=True)
        else:
            print(f"[INFO] Calibration loaded: focal={focal_length:.2f}px, person_height={REAL_PERSON_HEIGHT:.5f}m", flush=True)

        if display_available:
            try:
                cv2.namedWindow(VIDEO_WINDOW_NAME, cv2.WINDOW_NORMAL)
                print("[INFO] Live video window enabled. Press Q or Esc to quit.", flush=True)
            except cv2.error as exc:
                display_available = False
                print(f"[WARN] Cannot create OpenCV window ({exc}); running headless.", flush=True)
        else:
            print("[INFO] No desktop display; running with terminal reporting only.", flush=True)

        inference_future: Optional[Future] = None
        qr_future: Optional[Future] = None
        people: list[PersonDetection] = []
        track_labels: dict[int, str] = {}
        next_label = 1
        last_inference_success: Optional[float] = None
        last_inference_error: Optional[str] = None
        inference_submitted_at: Optional[float] = None
        inference_frame_time = 0.0       # capture time of the frame currently being inferred
        inference_latency_ms: Optional[float] = None
        recent_qr_values = deque(maxlen=QR_CONFIRMATION_WINDOW)
        scans_without_code = 0
        last_qr_type: Optional[str] = None
        last_qr_data: Optional[str] = None
        last_qr_boundary = None
        last_qr_sent: Optional[str] = None
        qr_reset_pending = False
        last_heartbeat = time.monotonic()
        last_commands: Optional[list[str]] = None
        last_terminal_report = 0.0
        previous_frame_time = time.perf_counter()
        fps = 0.0
        frame_count = 0
        last_frame_stamp = 0.0           # capture time of the last frame we processed

        last_non_empty_people_time: Optional[float] = None
        last_non_empty_status = "SAFE"
        loop_start_time = time.monotonic()
        auto_since = loop_start_time     # when human detection last (re)started
        previous_mode = "stopped"
        last_never_warning = 0.0
        camera_error_reported = False

        print(
            f"[START] Control loop ready; camera and vision model initializing in background "
            f"| Auto STOP <= {STOP_DISTANCE:.2f} m",
            flush=True,
        )
        log_monitor_header()
        while True:
            now = time.monotonic()
            if camera_init_future is not None and camera_init_future.done():
                try:
                    camera = camera_init_future.result()
                    camera_fourcc = int(camera.get(cv2.CAP_PROP_FOURCC))
                    camera_format = "".join(chr((camera_fourcc >> (8 * index)) & 0xFF) for index in range(4))
                    print(
                        f"[INFO] Camera opened: {FRAME_WIDTH}x{FRAME_HEIGHT} "
                        f"@ {camera.get(cv2.CAP_PROP_FPS):.1f} fps {camera_format}",
                        flush=True,
                    )
                    camera_capture = LatestFrameCapture(camera)
                    camera_capture.start()
                except (RuntimeError, OSError, cv2.error) as exc:
                    print(f"[ERROR] Camera initialization failed; Manual remains available, Auto will STOP: {exc}",
                          flush=True)
                camera_init_future = None

            if model_init_future is not None and model_init_future.done():
                try:
                    model, device = model_init_future.result()
                    print(f"[INFO] Vision model ready on {'CUDA' if device == 0 else 'CPU'}", flush=True)
                except (RuntimeError, ImportError, OSError) as exc:
                    model_error = str(exc)
                    print(f"[ERROR] Vision model initialization failed; Auto will STOP: {exc}", flush=True)
                model_init_future = None

            if qr_init_future is not None and qr_init_future.done():
                try:
                    qr_detector, marker_detector, contrast_enhancer = qr_init_future.result()
                except (RuntimeError, AttributeError, cv2.error) as exc:
                    print(f"[WARN] QR/ArUco initialization failed; QR output is unavailable: {exc}", flush=True)
                qr_init_future = None

            if camera_capture is not None:
                captured_frame, captured_at, camera_fps, camera_error = camera_capture.latest()
            else:
                captured_frame, captured_at, camera_fps = None, 0.0, 0.0
                camera_error = "camera is initializing or unavailable"
            camera_stale = captured_frame is None or now - captured_at > 1.0
            if camera_stale and not camera_error_reported:
                reason = camera_error or "camera frame is stale"
                print(f"[ERROR] {reason}; Auto is stopped while Manual remains available.", flush=True)
                camera_error_reported = True
            elif not camera_stale and camera_error_reported:
                print("[INFO] Camera frames recovered.", flush=True)
                camera_error_reported = False

            new_frame = not camera_stale and captured_at != last_frame_stamp
            if new_frame:
                last_frame_stamp = captured_at
                frame_age_ms = max(0.0, now - captured_at) * 1000.0
                frame_count += 1
                frame_time = time.perf_counter()
                frame_delta = frame_time - previous_frame_time
                previous_frame_time = frame_time
                if frame_delta > 1e-6:
                    fps = fps * 0.85 + (1.0 / frame_delta) * 0.15
            else:
                frame_age_ms = max(0.0, now - captured_at) * 1000.0 if captured_at else 0.0

            mode = dashboard.current_mode()
            manual = mode == "manual"
            if previous_mode == "manual" and not manual:
                # Leaving manual: throw away old detections so auto must see a FRESH result first.
                auto_since = now
                people = []
                last_inference_success = None
                last_inference_error = None
                last_non_empty_people_time = None
            previous_mode = mode

            # ---- collect finished work ----
            if inference_future is not None and inference_future.done():
                if inference_submitted_at is not None:
                    inference_latency_ms = max(0.0, now - inference_submitted_at) * 1000.0
                    inference_submitted_at = None
                try:
                    result_people, next_label = inference_future.result()
                    if inference_frame_time >= auto_since:      # ignore results from before auto resumed
                        people = result_people
                        last_inference_success = now
                        if last_inference_error is not None:
                            print(f"[VISION] inference recovered; cleared error: {last_inference_error}", flush=True)
                        last_inference_error = None
                except Exception as exc:
                    last_inference_error = str(exc)
                    print(f"[ERROR] YOLO inference failed; safety will STOP: {exc}", flush=True)
                inference_future = None

            if qr_future is not None and qr_future.done():
                try:
                    qr_data, detection_type, boundary = qr_future.result()
                    previous_qr = (last_qr_type, last_qr_data)
                    scans_without_code, last_qr_type, last_qr_data = update_qr_confirmation(
                        qr_data, detection_type, recent_qr_values, scans_without_code,
                        last_qr_type, last_qr_data,
                    )
                    if last_qr_data is not None and qr_data == last_qr_data and detection_type == last_qr_type:
                        last_qr_boundary = boundary
                    elif last_qr_data is None:
                        last_qr_boundary = None
                    current_qr = (last_qr_type, last_qr_data)
                    if current_qr != previous_qr and last_qr_data is not None:
                        print(f"[QR] Confirmed {last_qr_type}: {last_qr_data}", flush=True)
                    elif last_qr_data is None:
                        qr_reset_pending = qr_reset_pending or last_qr_sent is not None
                        last_qr_sent = None
                    if last_qr_data is not None:
                        qr_reset_pending = False
                except Exception as exc:
                    print(f"[WARN] QR/ArUco scan failed: {exc}", flush=True)
                qr_future = None

            # ---- start new work ----
            # YOLO runs back-to-back in auto/stopped, and NOT AT ALL in manual (saves CPU, faster video).
            if (new_frame and captured_frame is not None and inference_future is None
                    and not manual and model is not None):
                inference_submitted_at = now
                inference_frame_time = captured_at
                inference_future = executor.submit(
                    detect_people, model, captured_frame, focal_length,
                    device, track_labels, next_label,
                )
            if (new_frame and captured_frame is not None and qr_future is None
                    and frame_count % QR_SCAN_INTERVAL == 0
                    and qr_detector is not None and marker_detector is not None
                    and contrast_enhancer is not None):
                qr_future = executor.submit(
                    scan_qr_and_markers, captured_frame, frame_count,
                    qr_detector, marker_detector, contrast_enhancer,
                )

            frame = captured_frame.copy() if new_frame and captured_frame is not None else None

            # ------------------------------------------------------------
            # Safety decision
            # ------------------------------------------------------------
            if manual:
                # Manual: human detection / stopping logic is completely bypassed.
                people_view: list[PersonDetection] = []
                nearest = None
                status = "MANUAL"
                safety_command = "SAFE"
            else:
                people_view = people
                nearest = find_nearest_person(people)
                raw_status = get_robot_status(people, nearest)

                if people:
                    last_non_empty_people_time = now
                    last_non_empty_status = raw_status
                elif (last_non_empty_people_time is not None
                      and now - last_non_empty_people_time < SAFETY_HOLD_SECONDS):
                    raw_status = last_non_empty_status

                status = raw_status
                inference_age = None if last_inference_success is None else now - last_inference_success
                inference_is_stale = inference_age is None or inference_age > INFERENCE_STALE_SECONDS
                if (camera_stale or inference_is_stale or last_inference_error is not None
                        or model_error is not None):
                    status = "STOP"

                if (last_inference_success is None
                        and now - auto_since > STARTUP_GRACE_SECONDS
                        and now - last_never_warning >= 2.0):
                    print(f"[WARN] YOLO has produced NO result {now - auto_since:.1f}s after start; "
                          f"forcing STOP", flush=True)
                    last_never_warning = now
                safety_command = arduino_safety_command(status)

            drive_mode_command, left_speed, right_speed = dashboard.requested_drive(safety_command)

            # ------------------------------------------------------------
            # Serial: failure detection, reading, reconnect
            # ------------------------------------------------------------
            serial_failed = False
            if arduino is not None:
                if not arduino.is_open:
                    print("[SERIAL] port object reports closed", flush=True)
                    serial_failed = True
                elif arduino.port is None or not os.path.exists(arduino.port):
                    print(f"[SERIAL] {arduino.port} disappeared (USB drop or re-enumeration)", flush=True)
                    serial_failed = True

            if arduino is not None and not serial_failed:
                lines = read_arduino_lines(arduino, rx_buffer)
                if lines is None:
                    serial_failed = True
                else:
                    for line in lines:
                        if "MediCart ready" in line:
                            arduino_status = "READY"
                            boot_messages_seen += 1
                            if boot_messages_seen > 1:
                                print("[SERIAL] Arduino rebooted mid-session "
                                      "(brownout, USB glitch or DTR reset)", flush=True)
                        elif line.startswith("DBG obstacle="):
                            arduino_status = "OBSTACLE" if "obstacle=1" in line else "CLEAR"
                        elif line.startswith("DBG lineLost="):
                            arduino_status = "LINE_LOST" if "lineLost=1" in line else "LINE_OK"
                        elif line.startswith("DBG sensors="):
                            arduino_status = line.removeprefix("DBG sensors=")
                        elif line.startswith("DBG motor="):
                            try:
                                left_text, right_text = line.removeprefix("DBG motor=").split(",", 1)
                                arduino_left_speed = int(left_text)
                                arduino_right_speed = int(right_text)
                            except ValueError:
                                print(f"[WARN] Invalid motor feedback: {line}", flush=True)
                        else:
                            print(f"[ARDUINO] {line}", flush=True)

            if serial_failed:
                dashboard.emergency_stop()
                close_serial(arduino)
                arduino = None
                last_serial_attempt = now
                rx_buffer.clear()
                serial_failed = False

            if arduino is None and now - last_serial_attempt >= SERIAL_RETRY_INTERVAL_SECONDS:
                dashboard.emergency_stop()
                last_serial_attempt = now
                arduino = connect_serial()
                if arduino is not None:
                    serial_ready_at = now + SERIAL_BOOT_WAIT_SECONDS
                    boot_messages_seen = 0
                    rx_buffer.clear()
                    last_qr_sent = None
                    last_commands = None
                    arduino_status = "WAITING"
                    arduino_left_speed = 0
                    arduino_right_speed = 0
                    word = "reconnected" if had_connection else "connected"
                    print(f"[SERIAL] {word}, waiting {SERIAL_BOOT_WAIT_SECONDS:.1f}s for Mega boot", flush=True)
                    if last_port_name is not None and arduino.port != last_port_name:
                        print(f"[SERIAL] port changed {last_port_name} -> {arduino.port}", flush=True)
                    last_port_name = arduino.port
                    had_connection = True
                else:
                    serial_ready_at = 0.0

            port_open = arduino is not None and arduino.is_open
            serial_connected = port_open and now >= serial_ready_at
            if serial_connected:
                serial_state = "connected"
            elif port_open:
                serial_state = "booting"
            else:
                serial_state = "disconnected"

            # ------------------------------------------------------------
            # Commands: heartbeat every 100 ms, but a CHANGED command goes out immediately
            # (keypress, human detected, e-stop) instead of waiting for the next heartbeat.
            # ------------------------------------------------------------
            if serial_connected:
                commands = [drive_mode_command]
                if drive_mode_command == "MODE:MANUAL":
                    commands.append(f"DRIVE:{left_speed},{right_speed}")
                if commands != last_commands or heartbeat_due(now, last_heartbeat):
                    if all(send_command(arduino, command) for command in commands):
                        if drive_mode_command != (last_commands[0] if last_commands else None):
                            print(f"[CMD] {drive_mode_command}", flush=True)
                        last_commands = commands
                        last_heartbeat = now
                    else:
                        serial_failed = True

                if not serial_failed:
                    queued_qr_command = pending_qr_command(last_qr_data, last_qr_sent)
                    if queued_qr_command is not None:
                        if send_command(arduino, queued_qr_command):
                            last_qr_sent = last_qr_data
                        else:
                            serial_failed = True
                    elif qr_reset_pending:
                        if send_command(arduino, "RESET_QR"):
                            qr_reset_pending = False
                        else:
                            serial_failed = True

            if serial_failed:
                dashboard.emergency_stop()
                close_serial(arduino)
                arduino = None
                last_serial_attempt = now
                rx_buffer.clear()

            # ------------------------------------------------------------
            # Drawing
            # ------------------------------------------------------------
            if frame is not None:
                camera_point = draw_camera_reference(frame)
                for index, person in enumerate(people_view):
                    color = PERSON_COLORS[index % len(PERSON_COLORS)]
                    draw_distance_arrow(frame, camera_point, person, color)
                    draw_person_box(frame, person, person is nearest, color)
                    draw_person_keypoints(frame, person)
                draw_status(frame, people_view, nearest, status, camera_fps, focal_length)
                draw_qr_detection(frame, last_qr_boundary, last_qr_type, last_qr_data)

            if now - last_terminal_report >= TERMINAL_REPORT_INTERVAL_SECONDS:
                control_state = dashboard.control_snapshot()
                if drive_mode_command == "MODE:MANUAL":
                    monitor_mode = "MANUAL"
                elif drive_mode_command == "MODE:AUTO":
                    monitor_mode = "AUTO"
                else:
                    monitor_mode = "STOP"
                monitor_status = f"{status}/{serial_state}/{arduino_status}"
                if last_inference_error is not None:
                    monitor_status = "INFERENCE_ERROR"
                log_monitor_row(
                    now - loop_start_time, monitor_mode, str(control_state["keys"]),
                    arduino_left_speed, arduino_right_speed, camera_fps, frame_age_ms,
                    inference_latency_ms, monitor_status,
                )
                last_terminal_report = now

            if frame is None:
                dashboard.update_status(
                    status, len(people_view), nearest.distance if nearest is not None else None,
                    fps, serial_state, camera_fps, frame_age_ms, inference_latency_ms,
                    arduino_left_speed, arduino_right_speed,
                )
            key = -1
            if frame is not None:
                dashboard.publish_frame(
                    frame, status, len(people_view), nearest.distance if nearest is not None else None,
                    fps, serial_state, camera_fps, frame_age_ms, inference_latency_ms,
                    arduino_left_speed, arduino_right_speed,
                )
                if display_available:
                    try:
                        cv2.imshow(VIDEO_WINDOW_NAME, frame)
                        key = cv2.waitKey(1) & 0xFF
                    except cv2.error as exc:
                        display_available = False
                        print(f"[WARN] OpenCV display failed ({exc}); continuing headless.", flush=True)
            if key in (ord("q"), 27):
                break
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("[INFO] Vision stopped by user.", flush=True)
    finally:
        send_command(arduino, "STOP")
        dashboard.stop()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if camera_capture is not None:
            camera_capture.stop()
        elif camera is not None:
            camera.release()
        close_serial(arduino)
        if display_available:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
