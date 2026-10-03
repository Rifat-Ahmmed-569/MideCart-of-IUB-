"""
Live person detection and approximate monocular distance estimation.

Reconstructed from opencv_distance.cpython-312.pyc.
The original .py source is not embedded in a .pyc, so this is a functional
source reconstruction rather than a byte-for-byte recovery of the original.
"""

import json
import math
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import torch
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]

POSE_MODEL_PATH = PROJECT_DIRECTORY / "yolov8n-pose.pt"
FALLBACK_MODEL_PATH = PROJECT_DIRECTORY / "yolov8n.pt"
HAND_GESTURE_MODEL_PATH = Path(__file__).resolve().with_name(
    "gesture_recognizer.task"
)
HAND_GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
FACE_LANDMARKER_MODEL_PATH = Path(__file__).resolve().with_name(
    "face_landmarker.task"
)
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

INFERENCE_SIZE = 320
CPU_THREADS = 2
TERMINAL_REPORT_INTERVAL_SECONDS = 1.0

USE_CUDA = torch.cuda.is_available()

FEET_PER_METER = 3.28084
DEFAULT_PERSON_HEIGHT_METERS = 1.76784
PERSON_HEIGHT_METERS = DEFAULT_PERSON_HEIGHT_METERS

# This is replaced by calibration when a calibration file exists.
FOCAL_LENGTH_PIXELS = 0.0

STOP_DISTANCE = 2.0
WARNING_DISTANCE = 3.0
CONFIDENCE_THRESHOLD = 0.35
QR_RETRY_INTERVAL = 3
QR_LOST_FRAME_LIMIT = 10
QR_CONFIRMATION_WINDOW = 3
QR_VALID_VALUES = frozenset(str(value) for value in range(1, 11))

CALIBRATION_FILE = PROJECT_DIRECTORY / "camera_calibration.json"

PERSON_CLASS_ID = 0
WINDOW_NAME = "Robot Human Distance"

KEYPOINT_NAMES = (
    "Nose",
    "L-eye",
    "R-eye",
    "L-ear",
    "R-ear",
    "L-shoulder",
    "R-shoulder",
    "L-elbow",
    "R-elbow",
    "L-wrist",
    "R-wrist",
    "L-hip",
    "R-hip",
    "L-knee",
    "R-knee",
    "L-ankle",
    "R-ankle",
)

SKELETON_LINKS = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

FINGER_JOINTS = {
    "Thumb": (1, 2, 3, 4),
    "Index": (5, 6, 7, 8),
    "Middle": (9, 10, 11, 12),
    "Ring": (13, 14, 15, 16),
    "Pinky": (17, 18, 19, 20),
}

PERSON_COLORS = (
    (255, 180, 50),
    (60, 220, 120),
    (220, 120, 220),
    (80, 200, 240),
    (220, 180, 70),
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PersonDetection:
    person_id: str
    bbox: tuple
    center: tuple
    width: int
    height: int
    confidence: float
    distance: Optional[float]
    horizontal_position: str
    vertical_position: str
    partial: bool = False
    distance_from_partial_pose: Optional[float] = None
    keypoints: tuple = ()


# ---------------------------------------------------------------------------
# Camera / model
# ---------------------------------------------------------------------------

def initialize_camera():
    """Find a camera that opens and delivers a frame."""
    indices = list(dict.fromkeys((CAMERA_INDEX, 0, 1, 2)))
    backends = (
        (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
        if sys.platform == "win32"
        else (cv2.CAP_ANY,)
    )

    for index in indices:
        for backend in backends:
            camera = cv2.VideoCapture(index, backend)
            if not camera.isOpened():
                camera.release()
                continue

            camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            for _ in range(3):
                ok, _ = camera.read()
                if ok:
                    print(f"Using camera {index} (backend {backend}).")
                    return camera
                time.sleep(0.1)

            camera.release()

    raise RuntimeError(
        f"Could not open a working camera (tried indices: {indices}). "
        "Close other apps using the camera, check Windows camera permissions, "
        "or set CAMERA_INDEX near the top of this file."
    )


def initialize_qr_detectors():
    """Create QR and ArUco detectors with the scanner's tuned parameters."""
    aruco = cv2.aruco
    parameters = aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.02
    parameters.adaptiveThreshWinSizeMax = 33
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    marker_detector = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
        parameters,
    )
    qr_detector = cv2.QRCodeDetectorAruco()
    contrast_enhancer = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return qr_detector, marker_detector, contrast_enhancer


def scan_qr_and_markers(frame, frame_count, qr_detector, marker_detector, contrast_enhancer):
    """Detect supported ArUco IDs or QR payloads in the range 1 through 10."""
    corners, ids, _ = marker_detector.detectMarkers(frame)
    qr_data = None
    detection_type = "ARUCO MARKER"

    if ids is not None:
        for marker_id in ids.flatten():
            if 1 <= int(marker_id) <= 10:
                qr_data = str(int(marker_id))
                break

    if qr_data is None and frame_count % QR_RETRY_INTERVAL == 0:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced_frame = contrast_enhancer.apply(gray_frame)
        enlarged_frame = cv2.resize(
            enhanced_frame,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC,
        )
        retry_corners, retry_ids, _ = marker_detector.detectMarkers(enlarged_frame)

        if retry_ids is not None:
            for marker_id in retry_ids.flatten():
                if 1 <= int(marker_id) <= 10:
                    qr_data = str(int(marker_id))
                    corners = [corner / 1.5 for corner in retry_corners]
                    ids = retry_ids
                    break

    if qr_data is not None and ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    if qr_data is None:
        qr_text, qr_points, _ = qr_detector.detectAndDecode(frame)
        if qr_points is not None:
            cv2.polylines(
                frame,
                [qr_points.astype(int).reshape(-1, 1, 2)],
                True,
                (0, 255, 0),
                2,
            )

        if isinstance(qr_text, str) and qr_text.strip() in QR_VALID_VALUES:
            qr_data = qr_text.strip()
            detection_type = "QR CODE"

    return qr_data, detection_type


def update_qr_confirmation(
    qr_data,
    detection_type,
    recent_values,
    frames_without_marker,
    last_qr_type,
    last_qr_data,
):
    """Latch a QR/marker only after two matching frames and clear it after loss."""
    current_value = (detection_type, qr_data) if qr_data is not None else None
    recent_values.append(current_value)

    if qr_data is not None:
        frames_without_marker = 0
        if recent_values.count(current_value) >= 2:
            last_qr_type = detection_type
            last_qr_data = qr_data
    else:
        frames_without_marker += 1
        if frames_without_marker >= QR_LOST_FRAME_LIMIT:
            last_qr_type = None
            last_qr_data = None
            recent_values.clear()

    return frames_without_marker, last_qr_type, last_qr_data


def draw_detection_summary(frame, human_detected, qr_type, qr_data):
    """Show current human and stable QR/marker state in the camera view."""
    human_value = "YES" if human_detected else "NO"
    qr_value = "NONE" if qr_data is None else f"{qr_type} {qr_data}"
    label = f"HUMAN: {human_value} | QR: {qr_value}"
    height, width = frame.shape[:2]
    scale = 0.58
    thickness = 2
    text_size, _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    x = 12
    y = height - 108
    right = min(width - 4, x + text_size[0] + 16)

    cv2.rectangle(
        frame,
        (x, y - text_size[1] - 8),
        (right, y + 6),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x + 8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (70, 230, 100) if human_detected else (80, 100, 255),
        thickness,
        cv2.LINE_AA,
    )


def load_model():
    """Prefer pose weights; use the local detector if pose weights are unavailable."""
    if not USE_CUDA:
        torch.set_num_threads(CPU_THREADS)

    if POSE_MODEL_PATH.is_file():
        return YOLO(str(POSE_MODEL_PATH))

    if FALLBACK_MODEL_PATH.is_file():
        print(
            f"{POSE_MODEL_PATH} is not installed. "
            f"Using {FALLBACK_MODEL_PATH}; partial-body distances need pose "
            "weights and will remain unknown."
        )
        return YOLO(str(FALLBACK_MODEL_PATH))

    # Let Ultralytics download/use the default pose model if possible.
    try:
        return YOLO("yolov8n-pose.pt")
    except Exception:
        return YOLO(str(FALLBACK_MODEL_PATH))


def ensure_task_model(model_path, model_url):
    """Download a MediaPipe task model once, keeping incomplete downloads separate."""
    if not model_path.is_file():
        temporary_path = model_path.with_suffix(".task.download")
        try:
            urlretrieve(model_url, temporary_path)
            temporary_path.replace(model_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


def load_gesture_recognizer():
    """Create a video-mode recognizer for common static hand gestures."""
    ensure_task_model(HAND_GESTURE_MODEL_PATH, HAND_GESTURE_MODEL_URL)
    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(HAND_GESTURE_MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    return mp.tasks.vision.GestureRecognizer.create_from_options(options)


def load_face_landmarker():
    """Create a video-mode face tracker with blendshape expression scores."""
    ensure_task_model(FACE_LANDMARKER_MODEL_PATH, FACE_LANDMARKER_MODEL_URL)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=4,
        min_face_detection_confidence=0.55,
        min_face_presence_confidence=0.55,
        min_tracking_confidence=0.55,
        output_face_blendshapes=True,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def joint_angle_degrees(first, joint, last):
    """Return the angle formed by three MediaPipe landmarks at the middle one."""
    first_vector = (
        first.x - joint.x,
        first.y - joint.y,
        first.z - joint.z,
    )
    last_vector = (
        last.x - joint.x,
        last.y - joint.y,
        last.z - joint.z,
    )

    first_length = math.sqrt(sum(value * value for value in first_vector))
    last_length = math.sqrt(sum(value * value for value in last_vector))

    if first_length <= 1e-9 or last_length <= 1e-9:
        return 0.0

    cosine = sum(
        first_value * last_value
        for first_value, last_value in zip(first_vector, last_vector)
    ) / (first_length * last_length)

    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def classify_fingers(landmarks):
    """Classify each finger as extended using its two main joint angles."""
    states = {}

    for finger_name, (base, first_joint, second_joint, tip) in FINGER_JOINTS.items():
        first_angle = joint_angle_degrees(
            landmarks[base],
            landmarks[first_joint],
            landmarks[second_joint],
        )
        second_angle = joint_angle_degrees(
            landmarks[first_joint],
            landmarks[second_joint],
            landmarks[tip],
        )
        threshold = 150 if finger_name == "Thumb" else 155
        states[finger_name] = first_angle >= threshold and second_angle >= 145

    return states


def get_handedness_label(result, hand_index):
    """Return anatomical handedness for this unmirrored camera image."""
    if hand_index >= len(result.handedness) or not result.handedness[hand_index]:
        return "Hand"

    label = result.handedness[hand_index][0].category_name
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def get_gesture_classification(result, hand_index):
    """Return the top canned gesture and its score for one detected hand."""
    if hand_index >= len(result.gestures) or not result.gestures[hand_index]:
        return "Unrecognized", 0.0

    category = result.gestures[hand_index][0]
    name = category.category_name
    if name in ("", "None", "Unknown"):
        name = "Unrecognized"

    return name, float(category.score)


def summarize_face_expressions(face_result):
    """Extract visible expression cues and scores without inferring emotion."""
    summaries = []

    for face_index, categories in enumerate(face_result.face_blendshapes):
        scores = {
            category.category_name: float(category.score)
            for category in categories
        }
        left_blink = scores.get("eyeBlinkLeft", 0.0)
        right_blink = scores.get("eyeBlinkRight", 0.0)
        smile = statistics.mean(
            (scores.get("mouthSmileLeft", 0.0), scores.get("mouthSmileRight", 0.0))
        )
        frown = statistics.mean(
            (scores.get("mouthFrownLeft", 0.0), scores.get("mouthFrownRight", 0.0))
        )
        jaw_open = scores.get("jawOpen", 0.0)
        brow_raise = scores.get("browInnerUp", 0.0)

        cues = []
        if left_blink >= 0.65 and right_blink >= 0.65:
            cues.append("blink/eyes closed")
        elif left_blink >= 0.65:
            cues.append("left-eye blink")
        elif right_blink >= 0.65:
            cues.append("right-eye blink")
        if smile >= 0.5:
            cues.append("smile-like")
        if frown >= 0.5:
            cues.append("frown-like")
        if jaw_open >= 0.45:
            cues.append("mouth open")
        if brow_raise >= 0.45:
            cues.append("brows raised")
        if not cues:
            cues.append("no strong tracked cue")

        summaries.append(
            {
                "face_id": face_index + 1,
                "cues": cues,
                "scores": {
                    "smile": smile,
                    "frown": frown,
                    "jaw_open": jaw_open,
                    "left_blink": left_blink,
                    "right_blink": right_blink,
                    "brow_raise": brow_raise,
                },
            }
        )

    return summaries


# ---------------------------------------------------------------------------
# Formatting / calibration
# ---------------------------------------------------------------------------

def format_distance(distance):
    """Format a metric distance with its equivalent in feet."""
    if distance is None:
        return "N/A"

    return f"{distance:.2f} m / {distance * FEET_PER_METER:.2f} ft"


def format_meter_entry(value):
    """Show a meter-valued calibration entry in both units while it is typed."""
    if not value:
        return "..."

    try:
        return f"{format_distance(float(value))}"
    except ValueError:
        return f"{value} m"


def load_calibration():
    """Load a previously saved focal length, if a valid one exists."""
    global FOCAL_LENGTH_PIXELS, PERSON_HEIGHT_METERS

    if not CALIBRATION_FILE.is_file():
        return

    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        focal = float(data["focal_length_pixels"])
        person_height = float(
            data.get("known_height_meters", DEFAULT_PERSON_HEIGHT_METERS)
        )

        if focal > 0 and 0.5 <= person_height <= 2.8:
            FOCAL_LENGTH_PIXELS = focal
            PERSON_HEIGHT_METERS = person_height

    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        print(f"Ignoring invalid calibration file: {CALIBRATION_FILE}")


def calculate_distance(pixel_height, real_height, focal_length):
    """
    Estimate distance with:

        D = (real height * focal length) / pixel height
    """
    if pixel_height is None or pixel_height <= 0:
        return None

    if focal_length is None or focal_length <= 0:
        return None

    return (real_height * focal_length) / pixel_height


# ---------------------------------------------------------------------------
# Pose-based partial/full height estimation
# ---------------------------------------------------------------------------

def estimate_full_person_pixel_height(keypoints):
    """
    Infer approximate full stature from visible COCO pose-joint segments.

    Segment proportions are population averages, so this is a fallback
    estimate, not a physical measurement.

    Returns:
        (estimated_pixel_height, partial)
    """
    if not keypoints or len(keypoints) < 17:
        return None, True

    points = []
    for point in keypoints:
        if point is None or len(point) < 2:
            points.append(None)
            continue

        try:
            x, y = float(point[0]), float(point[1])
            confidence = float(point[2]) if len(point) > 2 else 1.0

            if confidence < 0.35:
                points.append(None)
            else:
                points.append((x, y))
        except (TypeError, ValueError):
            points.append(None)

    def joint(index):
        if index >= len(points):
            return None
        return points[index]

    def joint_midpoint(a, b):
        pa = joint(a)
        pb = joint(b)

        if pa is None or pb is None:
            return None

        return (
            (pa[0] + pb[0]) / 2.0,
            (pa[1] + pb[1]) / 2.0,
        )

    def distance(a, b):
        if a is None or b is None:
            return None

        return math.hypot(a[0] - b[0], a[1] - b[1])

    estimates = []

    # Approximate population proportions of total body height.
    # These are only used when the complete bounding-box height is unavailable.
    SEGMENT_PROPORTIONS = (
        ((5, 6), 0.12),    # shoulder width / rough upper-body scale
        ((5, 11), 0.29),   # left torso
        ((6, 12), 0.29),   # right torso
        ((11, 12), 0.18),  # hip width
        ((11, 13), 0.245), # left thigh
        ((12, 14), 0.245), # right thigh
        ((13, 15), 0.246), # left lower leg
        ((14, 16), 0.246), # right lower leg
    )

    for (a, b), proportion in SEGMENT_PROPORTIONS:
        if proportion <= 0:
            continue

        value = distance(joint(a), joint(b))
        if value is not None and value > 0:
            estimates.append(value / proportion)

    # Direct visible top-to-bottom estimates.
    nose = joint(0)
    shoulder_mid = joint_midpoint(5, 6)
    hip_mid = joint_midpoint(11, 12)
    knee_left = joint(13)
    knee_right = joint(14)
    ankle_left = joint(15)
    ankle_right = joint(16)

    head_to_hip = distance(nose, hip_mid)
    if head_to_hip is not None and head_to_hip > 0:
        estimates.append(head_to_hip / 0.535)

    shoulder_to_hip = distance(shoulder_mid, hip_mid)
    if shoulder_to_hip is not None and shoulder_to_hip > 0:
        estimates.append(shoulder_to_hip / 0.29)

    hip_to_ankle_values = [
        distance(hip_mid, ankle_left),
        distance(hip_mid, ankle_right),
    ]

    for value in hip_to_ankle_values:
        if value is not None and value > 0:
            estimates.append(value / 0.49)

    if not estimates:
        return None, True

    # Reject obviously unstable estimates caused by a few bad landmarks.
    pixel_height = statistics.median(estimates)

    if pixel_height <= 0:
        return None, True

    return pixel_height, True


# ---------------------------------------------------------------------------
# Position / detection
# ---------------------------------------------------------------------------

def calculate_position(center, frame_width, frame_height):
    """Describe the box center as left/center/right and top/center/bottom."""
    x, y = center

    horizontal = (
        "LEFT" if x < frame_width / 3
        else "RIGHT" if x > 2 * frame_width / 3
        else "CENTER"
    )

    vertical = (
        "TOP" if y < frame_height / 3
        else "BOTTOM" if y > 2 * frame_height / 3
        else "CENTER"
    )

    return horizontal, vertical


def detect_people(model, frame, focal_length=FOCAL_LENGTH_PIXELS):
    """Detect and track only people, returning independent data for each box."""
    if not USE_CUDA:
        torch.set_num_threads(CPU_THREADS)

    device = 0 if USE_CUDA else "cpu"

    results = model.track(
        frame,
        persist=True,
        classes=[PERSON_CLASS_ID],
        conf=CONFIDENCE_THRESHOLD,
        imgsz=INFERENCE_SIZE,
        device=device,
        max_det=20,
        verbose=False,
    )

    people = []

    if not results:
        return people

    result = results[0]

    if result.boxes is None:
        return people

    frame_height, frame_width = frame.shape[:2]

    boxes = result.boxes
    keypoints_obj = getattr(result, "keypoints", None)

    for index, box in enumerate(boxes):
        confidence = float(box.conf.item())
        class_id = int(box.cls.item())

        if class_id != PERSON_CLASS_ID:
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        touches_frame_edge = (
            x1 <= 1
            or y1 <= 1
            or x2 >= frame_width - 2
            or y2 >= frame_height - 2
        )

        x1 = max(0, min(frame_width - 1, x1))
        y1 = max(0, min(frame_height - 1, y1))
        x2 = max(0, min(frame_width - 1, x2))
        y2 = max(0, min(frame_height - 1, y2))

        width = int(round(max(0, x2 - x1)))
        height = int(round(max(0, y2 - y1)))

        center = (
            int(round((x1 + x2) / 2)),
            int(round((y1 + y2) / 2)),
        )

        track_id = getattr(box, "id", None)

        if track_id is not None:
            try:
                person_id = str(int(track_id[0].item()))
            except Exception:
                person_id = f"Person {index + 1}"
        else:
            person_id = f"Person {index + 1}"

        keypoints = ()

        if keypoints_obj is not None and index < len(keypoints_obj):
            try:
                data = keypoints_obj[index].data.detach().cpu().numpy()[0]
                keypoints = tuple(
                    tuple(float(v) for v in point)
                    for point in data
                )
            except Exception:
                keypoints = ()

        partial = touches_frame_edge
        distance_from_partial_pose = None

        if keypoints:
            visible = {
                point_index: (
                    len(point) >= 3 and float(point[2]) >= CONFIDENCE_THRESHOLD
                )
                for point_index, point in enumerate(keypoints)
            }
            has_head = any(visible.get(index, False) for index in range(5))
            has_ankle = any(visible.get(index, False) for index in (15, 16))
            partial = partial or not (has_head and has_ankle)

        distance = None

        if not partial and height > 0:
            distance = calculate_distance(
                float(height),
                PERSON_HEIGHT_METERS,
                focal_length,
            )

        horizontal_position, vertical_position = calculate_position(
            center,
            frame_width,
            frame_height,
        )

        people.append(
            PersonDetection(
                person_id=person_id,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                center=center,
                width=width,
                height=height,
                confidence=confidence,
                distance=distance,
                horizontal_position=horizontal_position,
                vertical_position=vertical_position,
                partial=partial,
                distance_from_partial_pose=distance_from_partial_pose,
                keypoints=keypoints,
            )
        )

    return people


def smooth_person_distances(people, distance_history):
    """Reduce frame-to-frame range jitter without carrying stale detections."""
    present_ids = set()

    for person in people:
        person_id = person.person_id
        present_ids.add(person_id)

        if person.distance is None:
            distance_history.pop(person_id, None)
            continue

        samples = distance_history.setdefault(person_id, [])
        samples.append(float(person.distance))
        del samples[:-5]
        person.distance = statistics.median(samples)

    for person_id in tuple(distance_history):
        if person_id not in present_ids:
            del distance_history[person_id]


def find_nearest_person(people):
    """Return the closest person with a valid calibrated distance."""
    valid = [
        person
        for person in people
        if person.distance is not None
        and float(person.distance) > 0
    ]

    if not valid:
        return None

    return min(
        valid,
        key=lambda person: float(person.distance),
        default=None,
    )


def get_robot_status(people, nearest):
    """Produce a simple status suitable for later robot/serial integration."""
    distances = [
        float(person.distance)
        for person in people
        if person.distance is not None and float(person.distance) > 0
    ]

    if not distances:
        return "UNKNOWN" if people else "SAFE"

    nearest_distance = min(distances)

    if nearest_distance <= STOP_DISTANCE:
        return "STOP"

    if any(person.distance is None for person in people):
        return "UNKNOWN"

    if nearest_distance <= WARNING_DISTANCE:
        return "WARNING"

    if people:
        return "PERSON_DETECTED"

    return "SAFE"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_camera_reference(frame):
    """Draw a camera marker and a schematic field-of-view guide."""
    height, width = frame.shape[:2]

    camera_x = width // 2
    camera_y = height - 24

    cv2.line(
        frame,
        (camera_x, camera_y),
        (int(width * 0.08), 0),
        (95, 95, 95),
        1,
    )

    cv2.line(
        frame,
        (camera_x, camera_y),
        (int(width * 0.92), 0),
        (95, 95, 95),
        1,
    )

    cv2.circle(
        frame,
        (camera_x, camera_y),
        7,
        (255, 255, 255),
        -1,
    )

    cv2.circle(
        frame,
        (camera_x, camera_y),
        10,
        (25, 25, 25),
        2,
    )

    cv2.putText(
        frame,
        "CAMERA",
        (camera_x + 13, camera_y + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_distance_arrow(frame, person):
    """Draw an independent camera-to-person sightline and distance label."""
    if person is None:
        return

    height, width = frame.shape[:2]

    camera_point = (width // 2, height - 24)
    center = tuple(map(int, person.center))

    cv2.arrowedLine(
        frame,
        camera_point,
        center,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.04,
    )

    if person.partial:
        if person.distance_from_partial_pose is None:
            label = f"{person.person_id}: partial, distance unknown"
        else:
            label = (
                f"{person.person_id}: partial -> "
                f"~{format_distance(person.distance_from_partial_pose)}"
            )
    elif person.distance is None:
        label = f"{person.person_id}: calibrate"
    else:
        label = f"{person.person_id}: ~{format_distance(person.distance)}"

    text_size, _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        2,
    )

    x = max(4, min(width - text_size[0] - 8, center[0] + 8))
    y = max(20, min(height - 8, center[1] - 8))

    cv2.rectangle(
        frame,
        (x - 4, y - text_size[1] - 8),
        (x + text_size[0] + 4, y + 4),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_person_box(frame, person, color):
    """Draw a person box and its readable identity, distance, position, and confidence."""
    x1, y1, x2, y2 = person.bbox

    if person.distance is not None and person.distance <= STOP_DISTANCE:
        box_color = (0, 0, 255)
    elif person.distance is not None and person.distance <= WARNING_DISTANCE:
        box_color = (0, 128, 255)
    else:
        box_color = color

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        box_color,
        2,
    )

    if person.partial:
        distance_text = "partial: distance unknown"
        if person.distance_from_partial_pose is not None:
            distance_text = f"~{format_distance(person.distance_from_partial_pose)}"
    elif FOCAL_LENGTH_PIXELS <= 0:
        distance_text = "uncalibrated"
    elif person.distance is None:
        distance_text = "N/A"
    else:
        distance_text = f"~{format_distance(person.distance)}"

    label = (
        f"{person.person_id}  {distance_text}  "
        f"{person.horizontal_position}/{person.vertical_position}  "
        f"{person.confidence:.0%}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2

    text_size, _ = cv2.getTextSize(
        label,
        font,
        scale,
        thickness,
    )

    label_x = max(0, min(x1, frame.shape[1] - text_size[0] - 8))
    label_y = max(text_size[1] + 8, y1)

    cv2.rectangle(
        frame,
        (label_x, label_y - text_size[1] - 8),
        (label_x + text_size[0] + 8, label_y + 4),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        frame,
        label,
        (label_x + 4, label_y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_person_keypoints(frame, keypoints):
    """Draw and label the visible COCO-17 body landmarks for a detected person."""
    if not keypoints:
        return

    visible = []

    for point in keypoints:
        if len(point) < 2:
            visible.append(None)
            continue

        try:
            x = int(round(point[0]))
            y = int(round(point[1]))
            confidence = float(point[2]) if len(point) > 2 else 1.0

            if confidence >= 0.35:
                visible.append((x, y))
            else:
                visible.append(None)
        except (TypeError, ValueError):
            visible.append(None)

    for a, b in SKELETON_LINKS:
        if a >= len(visible) or b >= len(visible):
            continue

        if visible[a] is not None and visible[b] is not None:
            cv2.line(
                frame,
                visible[a],
                visible[b],
                (40, 230, 230),
                2,
                cv2.LINE_AA,
            )

    for index, point in enumerate(visible):
        if point is None:
            continue

        cv2.circle(
            frame,
            point,
            4,
            (0, 40, 255),
            -1,
        )

        if index < len(KEYPOINT_NAMES):
            cv2.putText(
                frame,
                KEYPOINT_NAMES[index],
                (point[0] + 5, point[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def draw_hand_landmarks(frame, gesture_result):
    """Draw hand joints, finger states, and the top common gesture label."""
    if not gesture_result or not gesture_result.hand_landmarks:
        return

    frame_height, frame_width = frame.shape[:2]
    fingertip_to_name = {
        4: "Thumb",
        8: "Index",
        12: "Middle",
        16: "Ring",
        20: "Pinky",
    }

    for hand_index, landmarks in enumerate(gesture_result.hand_landmarks):
        states = classify_fingers(landmarks)
        points = [
            (
                max(0, min(frame_width - 1, int(round(landmark.x * frame_width)))),
                max(0, min(frame_height - 1, int(round(landmark.y * frame_height)))),
            )
            for landmark in landmarks
        ]

        handedness = get_handedness_label(gesture_result, hand_index)
        gesture_name, gesture_score = get_gesture_classification(
            gesture_result,
            hand_index,
        )

        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[start],
                points[end],
                (255, 210, 40),
                2,
                cv2.LINE_AA,
            )

        for landmark_index, point in enumerate(points):
            finger_name = fingertip_to_name.get(landmark_index)
            if finger_name is None:
                color = (255, 255, 255)
            elif states[finger_name]:
                color = (60, 230, 90)
            else:
                color = (0, 140, 255)

            cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)

        extended = [name for name, is_extended in states.items() if is_extended]
        label = (
            f"{handedness}: {gesture_name} {gesture_score:.0%}"
            f" | {len(extended)}/5"
        )
        if extended:
            label += ": " + ", ".join(extended)

        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            2,
        )
        min_x = min(point[0] for point in points)
        min_y = min(point[1] for point in points)
        label_x = max(4, min(frame_width - text_size[0] - 8, min_x))
        label_y = max(text_size[1] + 8, min_y - 8)

        cv2.rectangle(
            frame,
            (label_x - 4, label_y - text_size[1] - 6),
            (label_x + text_size[0] + 4, label_y + 4),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_face_expression_cues(frame, face_result, summaries):
    """Outline faces and show visible expression cues, not inferred emotions."""
    if not face_result or not face_result.face_landmarks:
        return

    frame_height, frame_width = frame.shape[:2]

    for face_index, landmarks in enumerate(face_result.face_landmarks):
        x_values = [landmark.x for landmark in landmarks]
        y_values = [landmark.y for landmark in landmarks]
        x1 = max(0, min(frame_width - 1, int(min(x_values) * frame_width)))
        y1 = max(0, min(frame_height - 1, int(min(y_values) * frame_height)))
        x2 = max(0, min(frame_width - 1, int(max(x_values) * frame_width)))
        y2 = max(0, min(frame_height - 1, int(max(y_values) * frame_height)))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 190, 40), 2)

        if face_index < len(summaries):
            cues = ", ".join(summaries[face_index]["cues"])
        else:
            cues = "no blendshape cues"
        label = f"Face {face_index + 1} cues: {cues}"

        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            2,
        )
        label_x = max(4, min(frame_width - text_size[0] - 8, x1))
        label_y = max(text_size[1] + 8, y1)

        cv2.rectangle(
            frame,
            (label_x - 4, label_y - text_size[1] - 6),
            (label_x + text_size[0] + 4, label_y + 4),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_status(frame, people, nearest, status, fps, focal_length):
    """Show frame-level information without covering the center of the image."""
    if nearest is None:
        nearest_text = "NEAREST PERSON: UNKNOWN"
        distance_text = "DISTANCE: N/A"
    else:
        nearest_text = f"NEAREST PERSON: {nearest.person_id}"
        distance = nearest.distance

        if distance is None:
            distance_text = "DISTANCE: N/A"
        else:
            distance_text = f"DISTANCE: ~{format_distance(distance)}"

    if focal_length and focal_length > 0:
        focal_text = f"FOCAL: {focal_length:.2f} px"
    else:
        focal_text = "FOCAL: not calibrated"

    lines = [
        nearest_text,
        distance_text,
        f"STATUS: {status}",
        f"PEOPLE: {len(people)}   FPS: {fps:.1f}",
        focal_text,
        f"HEIGHT MODEL: {PERSON_HEIGHT_METERS:.2f} m",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX

    for index, line in enumerate(lines):
        y = 24 + index * 28

        cv2.rectangle(
            frame,
            (10, y - 20),
            (340, y + 6),
            (20, 20, 20),
            -1,
        )

        cv2.putText(
            frame,
            line,
            (20, y),
            font,
            0.58,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )

    if status == "SAFE":
        status_color = (50, 220, 90)
    elif status == "WARNING":
        status_color = (0, 165, 255)
    elif status == "STOP":
        status_color = (0, 0, 255)
    else:
        status_color = (245, 245, 245)

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (20, 24),
        font,
        0.58,
        status_color,
        2,
        cv2.LINE_AA,
    )

    if not people:
        cv2.putText(
            frame,
            "No person detected",
            (20, frame.shape[0] - 28),
            font,
            0.68,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Terminal reporting / calibration UI
# ---------------------------------------------------------------------------

def print_person_data(people, nearest, status):
    """Print a readable, rate-limited summary of the current detections."""
    if not people:
        print(
            f"No person detected | nearest distance: N/A | status: {status}"
        )
        return

    for person in people:
        if person.distance is None:
            distance_text = "distance unknown"
        else:
            distance_text = f"~{format_distance(person.distance)}"

        if person.partial:
            distance_text += " (partial person; distance unavailable)"

        print(
            f"{person.person_id}: {distance_text}"
            f" | position={person.horizontal_position}/{person.vertical_position}"
            f" | confidence={person.confidence:.0%}"
        )

        if person.keypoints:
            visible = [
                KEYPOINT_NAMES[index]
                for index, point in enumerate(person.keypoints)
                if index < len(KEYPOINT_NAMES)
                and point is not None
                and len(point) >= 3
                and float(point[2]) >= 0.35
            ]

            print(
                "  visible landmarks: "
                + (", ".join(visible) if visible else "none")
            )
        else:
            print(
                "  body landmarks: unavailable "
                "(pose model weights not loaded)"
            )

    if nearest is None:
        print(f"Nearest person: unknown distance | status: {status}")
    else:
        distance_text = (
            format_distance(nearest.distance)
            if nearest.distance is not None
            else "unknown distance"
        )

        print(
            f"Nearest person: {nearest.person_id}"
            f" at {distance_text}"
            f" | status: {status}"
        )


def print_face_and_gesture_data(gesture_result, face_summaries):
    """Print transient hand gesture and facial movement summaries."""
    if gesture_result.hand_landmarks:
        for hand_index, landmarks in enumerate(gesture_result.hand_landmarks):
            handedness = get_handedness_label(gesture_result, hand_index)
            gesture_name, gesture_score = get_gesture_classification(
                gesture_result,
                hand_index,
            )
            states = classify_fingers(landmarks)
            extended = [name for name, is_extended in states.items() if is_extended]
            extended_text = ", ".join(extended) if extended else "none"
            print(
                f"{handedness} hand: gesture={gesture_name} "
                f"({gesture_score:.0%}) | extended={extended_text}"
            )
    else:
        print("No hands detected")

    if face_summaries:
        for summary in face_summaries:
            scores = summary["scores"]
            print(
                f"Face {summary['face_id']} visible cues (not emotion): "
                f"{', '.join(summary['cues'])} | "
                f"smile={scores['smile']:.2f}, "
                f"jaw={scores['jaw_open']:.2f}, "
                f"blinkL={scores['left_blink']:.2f}, "
                f"blinkR={scores['right_blink']:.2f}, "
                f"brow={scores['brow_raise']:.2f}"
            )
    else:
        print("No face detected")


def draw_calibration_prompt(frame, calibration):
    """Display the active calibration prompt in the camera window."""
    if not calibration.get("active"):
        return

    phase = calibration.get("phase")

    if phase == "height":
        value = format_meter_entry(calibration.get("height", ""))
        prompt = f"Person height (m): {value}  [Enter]"
    else:
        value = format_meter_entry(calibration.get("distance", ""))
        prompt = f"Camera-to-person distance (m): {value}  [Enter]"

    prompt += "   C: cancel"

    h, w = frame.shape[:2]
    text_size, _ = cv2.getTextSize(
        prompt,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        2,
    )
    box_width = min(w, max(620, text_size[0] + 36))

    cv2.rectangle(
        frame,
        (0, h - 86),
        (box_width, h),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        frame,
        prompt,
        (18, h - 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (80, 230, 230),
        2,
        cv2.LINE_AA,
    )

    message = calibration.get("message")
    if message:
        cv2.putText(
            frame,
            message,
            (18, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )


def calibration_mode(frame, calibration, current_pixel_height, key):
    """Handle nonblocking calibration keystrokes and persist a new focal length."""
    global FOCAL_LENGTH_PIXELS, PERSON_HEIGHT_METERS

    if not calibration.get("active"):
        return

    if key in (ord("c"), ord("C")):
        calibration.clear()
        return

    if key == 27:
        calibration.clear()
        return

    if key == 13:
        phase = calibration.get("phase")

        if phase == "height":
            try:
                height = float(calibration.get("height", ""))

                if not 0.5 <= height <= 2.8:
                    raise ValueError

                calibration.update(
                    {
                        "phase": "distance",
                        "height": height,
                        "distance": "",
                    }
                )
            except ValueError:
                calibration["message"] = "Enter a height from 0.5 to 2.8 m"

        elif phase == "distance":
            try:
                distance = float(calibration.get("distance", ""))

                if distance <= 0:
                    raise ValueError

                height = float(calibration["height"])

                if current_pixel_height is None or current_pixel_height <= 0:
                    calibration["message"] = (
                        "Show exactly one fully visible person, then press Enter"
                    )
                    return

                focal_length = (
                    float(current_pixel_height) * distance / height
                )

                if focal_length <= 0:
                    raise ValueError

                FOCAL_LENGTH_PIXELS = focal_length
                PERSON_HEIGHT_METERS = height

                try:
                    with open(CALIBRATION_FILE, "w", encoding="utf-8") as file:
                        json.dump(
                            {
                                "focal_length_pixels": focal_length,
                                "known_height_meters": height,
                                "known_distance_meters": distance,
                                "measured_pixel_height": current_pixel_height,
                            },
                            file,
                            indent=2,
                        )
                except OSError as exc:
                    print(f"Could not save calibration: {exc}")

                print(
                    f"Calibrated focal length: {focal_length:.2f} pixels; "
                    f"known height {format_distance(height)}, "
                    f"known distance {format_distance(distance)}"
                )

                calibration.clear()

            except (ValueError, TypeError, KeyError):
                calibration["message"] = "Enter a number greater than zero"

        return

    # Numeric entry for the current phase.
    if key in range(ord("0"), ord("9") + 1) or key in (ord("."), ord("-")):
        calibration.pop("message", None)
        calibration[calibration["phase"]] = (
            calibration.get(calibration["phase"], "") + chr(key)
        )

    elif key in (8, 127):
        calibration.pop("message", None)
        phase = calibration["phase"]
        calibration[phase] = calibration.get(phase, "")[:-1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run live person detection, distance estimation, and visualization."""
    global FOCAL_LENGTH_PIXELS, PERSON_HEIGHT_METERS

    try:
        camera = initialize_camera()
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(1)

    try:
        model = load_model()
    except Exception as exc:
        camera.release()
        print(
            f"Could not load model: {exc}\n"
            "Install the model weights or ensure they can be downloaded."
        )
        raise SystemExit(1)

    gesture_recognizer = None
    face_landmarker = None
    try:
        qr_detector, marker_detector, contrast_enhancer = initialize_qr_detectors()
        gesture_recognizer = load_gesture_recognizer()
        face_landmarker = load_face_landmarker()
    except Exception as exc:
        if gesture_recognizer is not None:
            gesture_recognizer.close()
        camera.release()
        print(
            f"Could not initialize face/gesture tracking: {exc}\n"
            "Check your internet connection for the first model downloads."
        )
        raise SystemExit(1)

    load_calibration()

    print(
        f"Camera ready with {model} at "
        f"{FRAME_WIDTH}x{FRAME_HEIGHT}px. "
        "Press Q to quit, C to calibrate, or R to clear saved calibration."
    )
    print("Click the camera window to type calibration values.")
    print(
        "Face output reports visible movement cues, not emotional states; "
        "hand output covers common gestures, not full sign-language translation. "
        "QR and ArUco scanning accepts values 1 through 10."
    )

    previous_time = time.perf_counter()
    last_terminal_report = 0.0
    fps = 0.0
    distance_history = {}
    frame_count = 0
    recent_qr_values = deque(maxlen=QR_CONFIRMATION_WINDOW)
    frames_without_marker = 0
    last_qr_data = None
    last_qr_type = None
    last_reported_detection_state = None

    calibration = {
        "active": False,
        "phase": "height",
        "height": "",
        "distance": "",
    }

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("Camera frame unavailable; stopping safely.")
                break

            frame_count += 1
            current_time = time.perf_counter()
            delta = current_time - previous_time
            previous_time = current_time

            if delta > 1e-6:
                instantaneous_fps = 1.0 / delta
                fps = fps * 0.85 + instantaneous_fps * 0.15

            people = detect_people(
                model,
                frame,
                FOCAL_LENGTH_PIXELS,
            )
            smooth_person_distances(people, distance_history)
            frame_timestamp_ms = int(current_time * 1000)
            media_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            gesture_result = gesture_recognizer.recognize_for_video(
                media_image,
                frame_timestamp_ms,
            )
            face_result = face_landmarker.detect_for_video(
                media_image,
                frame_timestamp_ms,
            )
            face_summaries = summarize_face_expressions(face_result)
            qr_data, detection_type = scan_qr_and_markers(
                frame,
                frame_count,
                qr_detector,
                marker_detector,
                contrast_enhancer,
            )
            (
                frames_without_marker,
                last_qr_type,
                last_qr_data,
            ) = update_qr_confirmation(
                qr_data,
                detection_type,
                recent_qr_values,
                frames_without_marker,
                last_qr_type,
                last_qr_data,
            )

            nearest = find_nearest_person(people)
            status = get_robot_status(people, nearest)
            human_detected = bool(people)

            detection_state = (human_detected, last_qr_type, last_qr_data)
            if detection_state != last_reported_detection_state:
                human_value = "YES" if human_detected else "NO"
                qr_value = (
                    "NONE"
                    if last_qr_data is None
                    else f"{last_qr_type} {last_qr_data}"
                )
                print(
                    f"DETECTED DATA: HUMAN={human_value} | QR={qr_value}",
                    flush=True,
                )
                last_reported_detection_state = detection_state

            draw_camera_reference(frame)

            for index, person in enumerate(people):
                color = PERSON_COLORS[index % len(PERSON_COLORS)]

                draw_person_box(frame, person, color)
                draw_person_keypoints(frame, person.keypoints)

            draw_hand_landmarks(frame, gesture_result)
            draw_face_expression_cues(frame, face_result, face_summaries)
            draw_distance_arrow(frame, nearest)
            draw_detection_summary(
                frame,
                human_detected,
                last_qr_type,
                last_qr_data,
            )

            if (
                current_time - last_terminal_report
                >= TERMINAL_REPORT_INTERVAL_SECONDS
            ):
                print_person_data(people, nearest, status)
                print_face_and_gesture_data(gesture_result, face_summaries)
                last_terminal_report = current_time

            draw_status(
                frame,
                people,
                nearest,
                status,
                fps,
                FOCAL_LENGTH_PIXELS,
            )

            draw_calibration_prompt(frame, calibration)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            if key in (ord("c"), ord("C")) and not calibration["active"]:
                calibration = {
                    "active": True,
                    "phase": "height",
                    "height": "",
                    "distance": "",
                }
                key = 255

            if key in (ord("r"), ord("R")):
                try:
                    CALIBRATION_FILE.unlink(missing_ok=True)
                    FOCAL_LENGTH_PIXELS = 0.0
                    PERSON_HEIGHT_METERS = DEFAULT_PERSON_HEIGHT_METERS
                    print(
                        "Saved calibration cleared; "
                        "press C to calibrate again."
                    )
                except OSError as exc:
                    print(f"Could not clear calibration: {exc}")

            if calibration["active"]:
                # Use the nearest person's visible/full pixel height for calibration.
                current_pixel_height = None

                if len(people) == 1:
                    person = people[0]

                    if person.height > 0 and not person.partial:
                        current_pixel_height = float(person.height)

                # Calibration mode uses nonblocking keyboard handling.
                calibration_mode(
                    frame,
                    calibration,
                    current_pixel_height,
                    key,
                )

    finally:
        gesture_recognizer.close()
        face_landmarker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
