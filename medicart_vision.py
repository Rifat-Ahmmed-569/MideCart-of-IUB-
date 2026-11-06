from __future__ import annotations

import threading
import time
from math import isfinite
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.serving import BaseWSGIServer, make_server


MAX_MANUAL_SPEED = 180
MANUAL_CONTROL_LEASE_SECONDS = 0.5
_VALID_MODES = {"auto", "manual"}


class WebDashboard:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._mode = "stopped"
        self._stop_token = 0
        self._speed = 110
        self._forward = 0.0
        self._turn = 0.0
        self._keys: tuple[str, ...] = ()
        self._last_control_time = 0.0
        self._vision_status = "STOP"
        self._people_count = 0
        self._nearest_distance: Optional[float] = None
        self._fps = 0.0
        self._camera_fps = 0.0
        self._frame_latency_ms: Optional[float] = None
        self._inference_ms: Optional[float] = None
        self._motor_left = 0
        self._motor_right = 0
        self._serial_state = "starting"
        self._server: Optional[BaseWSGIServer] = None
        self._thread: Optional[threading.Thread] = None

        self.app = Flask(__name__, template_folder="templates", static_folder="static")
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.get("/")
        def index():
            return render_template("dashboard.html")

        @self.app.get("/video_feed")
        def video_feed():
            return Response(
                self._video_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-store"},
            )

        @self.app.get("/api/status")
        def status():
            with self._lock:
                return jsonify({
                    "mode": self._mode,
                    "stop_token": self._stop_token,
                    "speed": self._speed,
                    "vision": self._vision_status,
                    "people": self._people_count,
                    "nearest_distance": self._nearest_distance,
                    "fps": round(self._fps, 1),
                    "camera_fps": round(self._camera_fps, 1),
                    "frame_latency_ms": self._frame_latency_ms,
                    "inference_ms": self._inference_ms,
                    "motor_left": self._motor_left,
                    "motor_right": self._motor_right,
                    "keys": "+".join(key.upper() for key in self._keys) or "---",
                    "serial": self._serial_state,
                    "max_speed": MAX_MANUAL_SPEED,
                })

        @self.app.post("/api/mode")
        def set_mode():
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload, dict):
                return jsonify({"error": "request body must be a JSON object"}), 400
            mode = payload.get("mode")
            if not isinstance(mode, str) or mode not in _VALID_MODES:
                return jsonify({"error": "mode must be auto or manual"}), 400
            try:
                stop_token = int(payload.get("stop_token"))
            except (TypeError, ValueError):
                return jsonify({"error": "stop_token is required"}), 400
            with self._lock:
                if stop_token != self._stop_token:
                    return jsonify({"error": "mode request predates emergency stop"}), 409
                self._mode = mode
                self._forward = 0.0
                self._turn = 0.0
                self._keys = ()
                self._last_control_time = 0.0
            return jsonify({"mode": mode})

        @self.app.post("/api/stop")
        def stop():
            stop_token = self.emergency_stop()
            return jsonify({"mode": "stopped", "stop_token": stop_token})

        @self.app.post("/api/speed")
        def set_speed():
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload, dict):
                return jsonify({"error": "request body must be a JSON object"}), 400
            speed_value = payload.get("speed")
            if isinstance(speed_value, bool) or (
                isinstance(speed_value, float) and not speed_value.is_integer()
            ):
                return jsonify({"error": "speed must be an integer"}), 400
            try:
                speed = int(speed_value)
            except (TypeError, ValueError):
                return jsonify({"error": "speed must be an integer"}), 400
            if not 0 <= speed <= MAX_MANUAL_SPEED:
                return jsonify({"error": f"speed must be between 0 and {MAX_MANUAL_SPEED}"}), 400
            with self._lock:
                self._speed = speed
            return jsonify({"speed": speed})

        @self.app.post("/api/control")
        def control():
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload, dict):
                return jsonify({"error": "request body must be a JSON object"}), 400
            try:
                forward = float(payload.get("forward", 0))
                turn = float(payload.get("turn", 0))
            except (TypeError, ValueError):
                return jsonify({"error": "forward and turn must be numeric"}), 400
            if not (isfinite(forward) and isfinite(turn)
                    and -1.0 <= forward <= 1.0 and -1.0 <= turn <= 1.0):
                return jsonify({"error": "forward and turn must be between -1 and 1"}), 400
            keys = payload.get("keys", [])
            valid_keys = {"up", "down", "left", "right"}
            if not isinstance(keys, list) or any(
                not isinstance(key, str) or key not in valid_keys for key in keys
            ):
                return jsonify({"error": "keys must contain only up, down, left, or right"}), 400
            with self._lock:
                if self._mode != "manual":
                    return jsonify({"error": "select manual mode before driving"}), 409
                self._forward = forward
                self._turn = turn
                self._keys = tuple(sorted(set(keys)))
                self._last_control_time = time.monotonic()
            return jsonify({"forward": forward, "turn": turn})

    def _video_frames(self):
        while True:
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
            if frame is None:
                time.sleep(0.1)
                continue
            encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if encoded:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            time.sleep(0.05)

    def publish_frame(
        self,
        frame: np.ndarray,
        vision_status: str,
        people_count: int,
        nearest_distance: Optional[float],
        fps: float,
        serial_state: str,
        camera_fps: float = 0.0,
        frame_latency_ms: Optional[float] = None,
        inference_ms: Optional[float] = None,
        motor_left: int = 0,
        motor_right: int = 0,
    ) -> None:
        with self._lock:
            self._frame = frame.copy()
            self._vision_status = vision_status
            self._people_count = people_count
            self._nearest_distance = nearest_distance
            self._fps = fps
            self._camera_fps = camera_fps
            self._frame_latency_ms = frame_latency_ms
            self._inference_ms = inference_ms
            self._motor_left = motor_left
            self._motor_right = motor_right
            self._serial_state = serial_state

    def requested_drive(self, safety_command: str) -> tuple[str, int, int]:
        with self._lock:
            mode = self._mode
            speed = self._speed
            forward = self._forward
            turn = self._turn
            last_control_time = self._last_control_time

        if mode == "stopped" or safety_command == "STOP":
            return "STOP", 0, 0
        if mode == "auto":
            return "MODE:AUTO", 0, 0

        if time.monotonic() - last_control_time > MANUAL_CONTROL_LEASE_SECONDS:
            return "MODE:MANUAL", 0, 0

        left = round((forward + turn) * speed)
        right = round((forward - turn) * speed)
        peak = max(abs(left), abs(right), MAX_MANUAL_SPEED)
        if peak > MAX_MANUAL_SPEED:
            scale = MAX_MANUAL_SPEED / peak
            left = round(left * scale)
            right = round(right * scale)
        return "MODE:MANUAL", left, right

    def control_snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "mode": self._mode,
                "keys": "+".join(key.upper() for key in self._keys) or "---",
            }

    def emergency_stop(self) -> int:
        with self._lock:
            self._stop_token += 1
            self._mode = "stopped"
            self._forward = 0.0
            self._turn = 0.0
            self._keys = ()
            self._last_control_time = 0.0
            return self._stop_token

    def start(self, host: str = "127.0.0.1", port: int = 5000) -> bool:
        try:
            self._server = make_server(host, port, self.app, threaded=True)
        except OSError as exc:
            print(f"[WEB] Could not start dashboard at http://{host}:{port}: {exc}", flush=True)
            return False
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[WEB] Dashboard available at http://{host}:{port}", flush=True)
        return True

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
