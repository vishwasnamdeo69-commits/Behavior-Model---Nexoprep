"""
AI Interview Behavior Analyzer — Phase 4
Upgraded: Advanced detection · Warning system · Graph reports · Scorecard · Statistics
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import time
import json
import os
import datetime
import math

from analyzer_api import start_analyzer_api_server


def _safe_log(message: str) -> None:
    """Print to console without crashing on Windows cp1252 encoding issues."""
    try:
        print(message)
    except UnicodeEncodeError:
        try:
            print(message.encode("ascii", errors="replace").decode("ascii"))
        except Exception:
            pass
    except Exception:
        pass

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    print("[WARN] matplotlib not found — graph reports disabled.")


# ══════════════════════════════════════════════════════════════════
#  CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════

SMOOTHING_WINDOW        = 18
MOVEMENT_HISTORY        = 25
STATE_SMOOTH_WINDOW     = 12
GAZE_SMOOTH_WINDOW      = 10

# Posture thresholds (composite angle)
SPINE_STRAIGHT_MAX      = 12.0
SPINE_SLIGHT_MAX        = 24.0
SPINE_SLOUCH_MAX        = 38.0
# Anything above SPINE_SLOUCH_MAX → Heavy Slouch

MOVEMENT_STABLE_MAX     = 0.0012
MOVEMENT_MODERATE_MAX   = 0.004
MOVEMENT_RESTLESS_MIN   = 0.007

EXP_SMOOTH_ALPHA        = 0.30          # lower = smoother
POSTURE_CALIB_ALPHA     = 0.05          # very slow drift for calibration

BLINK_THRESHOLD         = 0.21
SMILE_RATIO_THRESHOLD   = 3.3
EYE_CLOSED_RATIO        = 0.15          # EAR below this = closed
EYE_CLOSED_DURATION     = 2.5          # seconds before closure warning

GAZE_H_INNER            = 0.32         # gaze ratio bounds for "At Camera"
GAZE_H_OUTER            = 0.68
GAZE_V_INNER            = 0.28
GAZE_V_OUTER            = 0.72

INTERVIEW_DURATION      = 120           # seconds; 0 = unlimited
LOG_INTERVAL_SEC        = 1.0
REPORTS_DIR             = "reports"

GAZE_BREAK_DURATION     = 3.5
MOVEMENT_SPIKE_VAR      = 0.007
POSTURE_INSTABILITY_SEC = 5.0
EYE_CLOSURE_WARN_SEC    = 2.5
FACE_GONE_WARN_SEC      = 3.0

# Warning cooldowns (seconds before same warning fires again)
WARNING_COOLDOWN: Dict[str, float] = {
    "Prolonged Gaze Break":     8.0,
    "Repeated Downward Gaze":   10.0,
    "Posture Instability":      8.0,
    "Heavy Slouch":             10.0,
    "Leaning Forward":          8.0,
    "High Movement Variance":   6.0,
    "Restless Behavior":        8.0,
    "Eyes Closed":              5.0,
    "Face Not Visible":         5.0,
    "Low Eye Engagement":       12.0,
    "Distraction Detected":     10.0,
    "Nervous Movement":         8.0,
}

FONT           = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN    = (0, 220, 110)
COLOR_RED      = (0, 70, 230)
COLOR_YELLOW   = (0, 200, 245)
COLOR_WHITE    = (245, 245, 245)
COLOR_ACCENT   = (200, 160, 0)
COLOR_MUTED    = (120, 120, 150)
COLOR_PANEL_BG = (12, 12, 22)
COLOR_HEADER   = (25, 25, 45)
COLOR_DIVIDER  = (50, 50, 80)
COLOR_WARN     = (0, 100, 255)
COLOR_TIMER    = (180, 220, 255)
COLOR_ORANGE   = (0, 140, 255)
COLOR_CYAN     = (220, 200, 0)
COLOR_PURPLE   = (200, 80, 180)


# ══════════════════════════════════════════════════════════════════
#  MEDIAPIPE SETUP
# ══════════════════════════════════════════════════════════════════

mp_pose      = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands     = mp.solutions.hands
mp_drawing   = mp.solutions.drawing_utils

POSE = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
)

FACE = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
)

HANDS = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
)

LM = mp_pose.PoseLandmark


# ══════════════════════════════════════════════════════════════════
#  FACE MESH LANDMARK INDICES
# ══════════════════════════════════════════════════════════════════

LEFT_IRIS       = [474, 475, 476, 477]
RIGHT_IRIS      = [469, 470, 471, 472]
LEFT_EYE_TOP    = 159
LEFT_EYE_BOT    = 145
LEFT_EYE_LEFT   = 33
LEFT_EYE_RIGHT  = 133
RIGHT_EYE_TOP   = 386
RIGHT_EYE_BOT   = 374
RIGHT_EYE_LEFT  = 362
RIGHT_EYE_RIGHT = 263
MOUTH_LEFT      = 61
MOUTH_RIGHT     = 291
MOUTH_TOP       = 13
MOUTH_BOTTOM    = 14
NOSE_TIP        = 1
CHIN            = 152
LEFT_CHEEK      = 234
RIGHT_CHEEK     = 454
FOREHEAD        = 10
LEFT_BROW_INNER = 336
RIGHT_BROW_INNER= 107


# ══════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class FrameLog:
    time:              float
    posture:           str
    movement:          str
    gaze:              str
    expression:        str
    gesture:           str
    posture_score:     float
    movement_score:    float
    eye_score:         float
    expression_score:  float
    gesture_score:     float
    composite_angle:   float
    variance:          float
    gaze_attention:    float   # 0-1 continuous attention estimate
    eye_ar:            float   # eye aspect ratio
    engagement_score:  float   # composite engagement 0-10
    attention_score:   float   # attention/focus score 0-10

    def to_dict(self):
        return {
            "time":             round(self.time, 2),
            "posture":          self.posture,
            "movement":         self.movement,
            "gaze":             self.gaze,
            "expression":       self.expression,
            "gesture":          self.gesture,
            "posture_score":    self.posture_score,
            "movement_score":   self.movement_score,
            "eye_score":        self.eye_score,
            "expression_score": self.expression_score,
            "gesture_score":    self.gesture_score,
            "composite_angle":  round(self.composite_angle, 2),
            "variance":         round(self.variance, 6),
            "gaze_attention":   round(self.gaze_attention, 3),
            "eye_ar":           round(self.eye_ar, 3),
            "engagement_score": self.engagement_score,
            "attention_score":  self.attention_score,
        }


@dataclass
class BehavioralEvent:
    time:       float
    event_type: str
    detail:     str
    severity:   str = "medium"   # low / medium / high

    def to_dict(self):
        return {
            "time":       round(self.time, 2),
            "event_type": self.event_type,
            "detail":     self.detail,
            "severity":   self.severity,
        }


# ══════════════════════════════════════════════════════════════════
#  STATE & SCORE SMOOTHING
# ══════════════════════════════════════════════════════════════════

class StateBuffer:
    """Majority-vote smoothing for categorical states."""
    def __init__(self, window=STATE_SMOOTH_WINDOW, default="Calibrating"):
        self.buf     = deque(maxlen=window)
        self.default = default

    def update(self, state: str) -> str:
        self.buf.append(state)
        if not self.buf:
            return self.default
        counts: Dict[str, int] = {}
        for s in self.buf:
            counts[s] = counts.get(s, 0) + 1
        return max(counts, key=counts.get)


class ScoreBuffer:
    """Exponential-weighted moving average for continuous scores."""
    def __init__(self, window=SMOOTHING_WINDOW):
        self.buf = deque(maxlen=window)

    def update(self, val: float) -> float:
        self.buf.append(val)
        return round(float(np.mean(self.buf)), 1) if self.buf else val

    def std(self) -> float:
        return float(np.std(self.buf)) if len(self.buf) > 1 else 0.0


class GazeBuffer:
    """Dedicated smoothing for gaze ratio values (horizontal + vertical)."""
    def __init__(self, window=GAZE_SMOOTH_WINDOW):
        self.h_buf = deque(maxlen=window)
        self.v_buf = deque(maxlen=window)

    def update(self, h: float, v: float) -> Tuple[float, float]:
        self.h_buf.append(h)
        self.v_buf.append(v)
        sh = float(np.median(self.h_buf))
        sv = float(np.median(self.v_buf))
        return sh, sv


class PostureCalibrator:
    """
    Adaptive baseline calibration.
    Computes a running reference composite angle during the first
    CALIB_FRAMES frames, then adjusts thresholds per-person.
    """
    CALIB_FRAMES = 60

    def __init__(self):
        self._samples: List[float] = []
        self._baseline: Optional[float] = None
        self._offset: float = 0.0
        self.calibrated: bool = False

    def feed(self, composite: float):
        if self.calibrated:
            return
        if composite > 0:
            self._samples.append(composite)
        if len(self._samples) >= self.CALIB_FRAMES:
            self._baseline = float(np.percentile(self._samples, 25))
            self._offset   = max(self._baseline - 8.0, 0.0)
            self.calibrated = True

    def adjust(self, composite: float) -> float:
        """Return composite with personal baseline removed."""
        return max(composite - self._offset, 0.0)

    @property
    def progress(self) -> float:
        return min(len(self._samples) / self.CALIB_FRAMES, 1.0)


# ══════════════════════════════════════════════════════════════════
#  SESSION LOGGER
# ══════════════════════════════════════════════════════════════════

class SessionLogger:
    def __init__(self):
        self.logs:       List[FrameLog] = []
        self.last_log_t: float          = -999.0

    def try_log(self, elapsed: float, frame_log: FrameLog):
        if elapsed - self.last_log_t >= LOG_INTERVAL_SEC:
            self.logs.append(frame_log)
            self.last_log_t = elapsed

    def get_logs(self) -> List[FrameLog]:
        return self.logs


# ══════════════════════════════════════════════════════════════════
#  TIMELINE TRACKER
# ══════════════════════════════════════════════════════════════════

class TimelineTracker:
    def __init__(self):
        self.times:       List[float] = []
        self.posture:     List[float] = []
        self.movement:    List[float] = []
        self.eye:         List[float] = []
        self.expression:  List[float] = []
        self.gesture:     List[float] = []
        self.overall:     List[float] = []
        self.engagement:  List[float] = []
        self.attention:   List[float] = []

    def record(self, elapsed: float, sp, sm, se, sx, sg, eng, att):
        self.times.append(round(elapsed, 2))
        self.posture.append(sp)
        self.movement.append(sm)
        self.eye.append(se)
        self.expression.append(sx)
        self.gesture.append(sg)
        self.overall.append(round((sp + sm + se + sx + sg) / 5, 2))
        self.engagement.append(eng)
        self.attention.append(att)

    def to_dict(self):
        return {
            "times":      self.times,
            "posture":    self.posture,
            "movement":   self.movement,
            "eye":        self.eye,
            "expression": self.expression,
            "gesture":    self.gesture,
            "overall":    self.overall,
            "engagement": self.engagement,
            "attention":  self.attention,
        }

    def trend(self, arr: List[float]) -> str:
        if len(arr) < 6:
            return "insufficient data"
        mid         = len(arr) // 2
        first_half  = float(np.mean(arr[:mid]))
        second_half = float(np.mean(arr[mid:]))
        delta       = second_half - first_half
        if delta > 0.5:
            return "improving"
        elif delta < -0.5:
            return "declining"
        return "stable"


# ══════════════════════════════════════════════════════════════════
#  BEHAVIORAL WARNING MANAGER
# ══════════════════════════════════════════════════════════════════

class BehavioralWarningManager:
    """
    Manages real-time warnings with per-type cooldown timers.
    Prevents spam, ensures warnings persist briefly on-screen.
    """

    def __init__(self):
        self._last_fired:    Dict[str, float] = {}
        self._active_warn:   Optional[str]    = None
        self._warn_expires:  float            = 0.0
        self.warn_log:       List[Dict]       = []   # full history

    def try_fire(self, warn_type: str, detail: str,
                 now: float, severity: str = "medium") -> bool:
        cooldown = WARNING_COOLDOWN.get(warn_type, 8.0)
        last     = self._last_fired.get(warn_type, -9999.0)
        if now - last < cooldown:
            return False
        self._last_fired[warn_type] = now
        self._active_warn  = warn_type
        self._warn_expires = now + 4.5
        self.warn_log.append({
            "time":     round(now, 2),
            "type":     warn_type,
            "detail":   detail,
            "severity": severity,
        })
        return True

    def current_warning(self, now: float) -> Optional[str]:
        if self._active_warn and now < self._warn_expires:
            return self._active_warn
        return None

    def warning_severity(self, warn_type: str) -> str:
        """Return severity for color coding."""
        high = {"Eyes Closed", "Face Not Visible", "Heavy Slouch",
                "Restless Behavior", "Nervous Movement"}
        low  = {"Low Eye Engagement", "Distraction Detected"}
        if warn_type in high:
            return "high"
        if warn_type in low:
            return "low"
        return "medium"


# ══════════════════════════════════════════════════════════════════
#  EVENT DETECTOR  (upgraded)
# ══════════════════════════════════════════════════════════════════

class EventDetector:
    def __init__(self):
        self.events:              List[BehavioralEvent] = []
        self.warning_manager:     BehavioralWarningManager = BehavioralWarningManager()
        self._gaze_away_start:    Optional[float] = None
        self._downward_gaze_buf:  deque = deque(maxlen=30)
        self._posture_bad_start:  Optional[float] = None
        self._eye_closed_start:   Optional[float] = None
        self._face_gone_start:    Optional[float] = None
        self._last_spike_t:       float = -999.0
        self._restless_start:     Optional[float] = None
        self._distraction_count:  int   = 0
        self._downward_count:     int   = 0

    def update(self, elapsed: float, gaze: str, posture: str,
               variance: float, eye_ar: float, face_visible: bool,
               movement: str):
        """Full per-frame event evaluation."""

        # ── Gaze away ──
        if gaze != "At Camera":
            if self._gaze_away_start is None:
                self._gaze_away_start = elapsed
            else:
                away_dur = elapsed - self._gaze_away_start
                if away_dur >= GAZE_BREAK_DURATION:
                    fired = self.warning_manager.try_fire(
                        "Prolonged Gaze Break",
                        f"Gaze away ({gaze}) for {away_dur:.1f}s",
                        elapsed, severity="medium")
                    if fired:
                        self._log(elapsed, "Prolonged Gaze Break",
                                  f"Gaze ({gaze}) for >{GAZE_BREAK_DURATION:.0f}s",
                                  "medium")
                        self._distraction_count += 1
                        self._gaze_away_start = elapsed
        else:
            self._gaze_away_start = None

        # ── Repeated downward gaze ──
        self._downward_gaze_buf.append(1 if gaze == "Looking Down" else 0)
        if len(self._downward_gaze_buf) == 30:
            down_ratio = sum(self._downward_gaze_buf) / 30
            if down_ratio > 0.55:
                self._downward_count += 1
                fired = self.warning_manager.try_fire(
                    "Repeated Downward Gaze",
                    f"Downward gaze {down_ratio*100:.0f}% of recent frames",
                    elapsed, severity="medium")
                if fired:
                    self._log(elapsed, "Repeated Downward Gaze",
                              f"Downward gaze ratio: {down_ratio:.2f}", "medium")

        # ── Posture ──
        if posture not in ("Straight", "Calibrating", "No Pose", "Unknown"):
            if self._posture_bad_start is None:
                self._posture_bad_start = elapsed
            else:
                bad_dur = elapsed - self._posture_bad_start
                if bad_dur >= POSTURE_INSTABILITY_SEC:
                    sev = "high" if posture == "Heavy Slouch" else "medium"
                    warn_key = "Heavy Slouch" if posture == "Heavy Slouch" else "Posture Instability"
                    fired = self.warning_manager.try_fire(
                        warn_key,
                        f"Posture ({posture}) sustained {bad_dur:.1f}s",
                        elapsed, severity=sev)
                    if fired:
                        self._log(elapsed, warn_key,
                                  f"Non-upright posture ({posture}) for {bad_dur:.1f}s", sev)
                        self._posture_bad_start = elapsed
        else:
            self._posture_bad_start = None

        # ── Movement spike ──
        if variance > MOVEMENT_SPIKE_VAR and elapsed - self._last_spike_t > 5.0:
            fired = self.warning_manager.try_fire(
                "High Movement Variance",
                f"Movement variance={variance:.5f}",
                elapsed, severity="medium")
            if fired:
                self._log(elapsed, "High Movement Variance",
                          f"Variance={variance:.5f}", "medium")
                self._last_spike_t = elapsed

        # ── Restless movement ──
        if movement == "Restless":
            if self._restless_start is None:
                self._restless_start = elapsed
            elif elapsed - self._restless_start >= 6.0:
                fired = self.warning_manager.try_fire(
                    "Restless Behavior",
                    f"Continuous restless movement {elapsed - self._restless_start:.1f}s",
                    elapsed, severity="high")
                if fired:
                    self._log(elapsed, "Restless Behavior",
                              "Sustained elevated movement", "high")
                    self._restless_start = elapsed
        else:
            self._restless_start = None

        # ── Eye closure ──
        if eye_ar < EYE_CLOSED_RATIO:
            if self._eye_closed_start is None:
                self._eye_closed_start = elapsed
            elif elapsed - self._eye_closed_start >= EYE_CLOSURE_WARN_SEC:
                fired = self.warning_manager.try_fire(
                    "Eyes Closed",
                    f"Eyes closed for {elapsed - self._eye_closed_start:.1f}s",
                    elapsed, severity="high")
                if fired:
                    self._log(elapsed, "Eyes Closed",
                              f"Prolonged eye closure detected", "high")
                    self._eye_closed_start = elapsed
        else:
            self._eye_closed_start = None

        # ── Face not visible ──
        if not face_visible:
            if self._face_gone_start is None:
                self._face_gone_start = elapsed
            elif elapsed - self._face_gone_start >= FACE_GONE_WARN_SEC:
                fired = self.warning_manager.try_fire(
                    "Face Not Visible",
                    f"Face absent for {elapsed - self._face_gone_start:.1f}s",
                    elapsed, severity="high")
                if fired:
                    self._log(elapsed, "Face Not Visible",
                              "Face landmarks lost", "high")
                    self._face_gone_start = elapsed
        else:
            self._face_gone_start = None

    def _log(self, t: float, etype: str, detail: str, severity: str):
        self.events.append(BehavioralEvent(
            time=t, event_type=etype, detail=detail, severity=severity))

    def get_events(self) -> List[BehavioralEvent]:
        return self.events

    def latest_warning(self, now: float = 0.0) -> Optional[str]:
        return self.warning_manager.current_warning(now)

    @property
    def distraction_count(self) -> int:
        return self._distraction_count

    @property
    def downward_count(self) -> int:
        return self._downward_count


# ══════════════════════════════════════════════════════════════════
#  BEHAVIOR STATISTICS CLASS
# ══════════════════════════════════════════════════════════════════

class BehaviorStatistics:
    """
    Computes analytical statistics from session log data.
    Provides averages, standard deviations, variance, stability metrics.
    """

    @staticmethod
    def compute_series_stats(data: List[float]) -> Dict:
        if not data:
            return {"mean": 0, "std": 0, "min": 0, "max": 0,
                    "variance": 0, "volatility": 0, "stability_pct": 0}
        arr  = np.array(data, dtype=np.float64)
        mean = float(np.mean(arr))
        std  = float(np.std(arr))
        var  = float(np.var(arr))
        # volatility = normalized std relative to range
        rng  = float(np.ptp(arr))
        vol  = (std / max(rng, 0.1)) * 100
        # stability = % of time within 1 std of mean
        in_range = np.sum(np.abs(arr - mean) <= std)
        stab_pct = float(in_range / len(arr) * 100)
        return {
            "mean":          round(mean, 3),
            "std":           round(std,  3),
            "min":           round(float(np.min(arr)), 3),
            "max":           round(float(np.max(arr)), 3),
            "variance":      round(var,  5),
            "volatility":    round(vol,  2),
            "stability_pct": round(stab_pct, 1),
        }

    @classmethod
    def from_logs(cls, logs: List[FrameLog]) -> Dict:
        if not logs:
            return {}
        return {
            "posture":    cls.compute_series_stats([l.posture_score    for l in logs]),
            "movement":   cls.compute_series_stats([l.movement_score   for l in logs]),
            "eye":        cls.compute_series_stats([l.eye_score        for l in logs]),
            "expression": cls.compute_series_stats([l.expression_score for l in logs]),
            "gesture":    cls.compute_series_stats([l.gesture_score    for l in logs]),
            "engagement": cls.compute_series_stats([l.engagement_score for l in logs]),
            "attention":  cls.compute_series_stats([l.attention_score  for l in logs]),
            "gaze_attention": cls.compute_series_stats([l.gaze_attention for l in logs]),
            "composite_angle":cls.compute_series_stats([l.composite_angle for l in logs]),
            "variance":   cls.compute_series_stats([l.variance for l in logs]),
        }


# ══════════════════════════════════════════════════════════════════
#  SESSION AGGREGATION ENGINE
# ══════════════════════════════════════════════════════════════════

class SessionAggregator:

    @staticmethod
    def compute(logs: List[FrameLog], timeline: TimelineTracker,
                duration: float, events: List[BehavioralEvent]) -> dict:
        if not logs:
            return {}

        n = len(logs)
        stats = BehaviorStatistics.from_logs(logs)

        avg_posture    = stats["posture"]["mean"]
        avg_movement   = stats["movement"]["mean"]
        avg_eye        = stats["eye"]["mean"]
        avg_expression = stats["expression"]["mean"]
        avg_gesture    = stats["gesture"]["mean"]
        avg_engagement = stats["engagement"]["mean"]
        avg_attention  = stats["attention"]["mean"]
        avg_overall    = round((avg_posture + avg_movement + avg_eye +
                                avg_expression + avg_gesture) / 5, 2)

        pct_straight  = round(sum(1 for l in logs if l.posture    == "Straight")  / n * 100, 1)
        pct_at_camera = round(sum(1 for l in logs if l.gaze       == "At Camera") / n * 100, 1)
        pct_stable    = round(sum(1 for l in logs if l.movement   == "Stable")    / n * 100, 1)
        pct_restless  = round(sum(1 for l in logs if l.movement   == "Restless")  / n * 100, 1)
        pct_smiling   = round(sum(1 for l in logs if l.expression == "Smiling")   / n * 100, 1)
        pct_neutral   = round(sum(1 for l in logs if l.expression == "Neutral")   / n * 100, 1)
        pct_open_palm = round(sum(1 for l in logs if "Open Palm" in l.gesture)    / n * 100, 1)
        pct_no_hands  = round(sum(1 for l in logs if l.gesture    == "No Hands")  / n * 100, 1)
        pct_down_gaze = round(sum(1 for l in logs if l.gaze == "Looking Down")    / n * 100, 1)
        avg_spine     = stats["composite_angle"]["mean"]
        avg_gaze_att  = stats["gaze_attention"]["mean"]

        event_counts: Dict[str, int] = {}
        for ev in events:
            event_counts[ev.event_type] = event_counts.get(ev.event_type, 0) + 1

        all_scores = {
            "posture":    avg_posture,
            "movement":   avg_movement,
            "eye_contact":avg_eye,
            "expression": avg_expression,
            "gesture":    avg_gesture,
        }
        strongest = max(all_scores, key=all_scores.get)
        weakest   = min(all_scores, key=all_scores.get)

        # Behavioral presence score (composite with engagement weight)
        behavioral_presence = round(
            avg_overall * 0.6 + avg_engagement * 0.25 + avg_attention * 0.15, 2)

        return {
            "duration_seconds":    round(duration, 1),
            "total_log_entries":   n,
            "avg_scores": {
                "posture":    round(avg_posture,    2),
                "movement":   round(avg_movement,   2),
                "eye_contact":round(avg_eye,        2),
                "expression": round(avg_expression, 2),
                "gesture":    round(avg_gesture,    2),
                "overall":    avg_overall,
                "engagement": round(avg_engagement, 2),
                "attention":  round(avg_attention,  2),
            },
            "percentages": {
                "straight_posture_pct":   pct_straight,
                "at_camera_gaze_pct":     pct_at_camera,
                "stable_movement_pct":    pct_stable,
                "restless_movement_pct":  pct_restless,
                "smiling_pct":            pct_smiling,
                "neutral_expression_pct": pct_neutral,
                "open_palm_gesture_pct":  pct_open_palm,
                "no_hands_visible_pct":   pct_no_hands,
                "downward_gaze_pct":      pct_down_gaze,
                "avg_gaze_attention":     round(avg_gaze_att, 3),
            },
            "statistics":           stats,
            "avg_spine_angle":      avg_spine,
            "strongest_signal":     strongest,
            "weakest_signal":       weakest,
            "behavioral_presence":  behavioral_presence,
            "event_counts":         event_counts,
            "total_events":         len(events),
            "trends": {
                "posture":    timeline.trend(timeline.posture),
                "movement":   timeline.trend(timeline.movement),
                "eye":        timeline.trend(timeline.eye),
                "overall":    timeline.trend(timeline.overall),
                "engagement": timeline.trend(timeline.engagement),
                "attention":  timeline.trend(timeline.attention),
            },
        }


# ══════════════════════════════════════════════════════════════════
#  GRAPH GENERATOR (matplotlib)
# ══════════════════════════════════════════════════════════════════

class GraphGenerator:
    """
    Generates professional dark-theme PNG graphs from session data.
    Saved automatically into reports/session_<ts>/
    """

    DARK_BG    = "#0d0d1a"
    GRID_COLOR = "#2a2a40"
    LINE_COLORS = {
        "overall":    "#00e87a",
        "posture":    "#00c8f5",
        "eye":        "#f5a623",
        "movement":   "#e052e0",
        "engagement": "#52d4e0",
        "attention":  "#e07852",
    }

    @classmethod
    def generate_all(cls, timeline: TimelineTracker, agg: dict,
                     events: List[BehavioralEvent], out_dir: str):
        if not MATPLOTLIB_OK:
            print("[WARN] matplotlib unavailable — skipping graphs.")
            return []

        os.makedirs(out_dir, exist_ok=True)
        paths = []

        paths.append(cls._timeline_dashboard(timeline, agg, out_dir))
        paths.append(cls._posture_detail(timeline, agg, out_dir))
        paths.append(cls._gaze_movement(timeline, agg, out_dir))
        paths.append(cls._engagement_attention(timeline, agg, out_dir))
        paths.append(cls._event_timeline(events, timeline, out_dir))
        paths.append(cls._scorecard_radar(agg, out_dir))

        return [p for p in paths if p]

    @classmethod
    def _setup_ax(cls, ax, title: str):
        ax.set_facecolor(cls.DARK_BG)
        ax.set_title(title, color="#ccccdd", fontsize=10, pad=8, fontweight="bold")
        ax.tick_params(colors="#888899", labelsize=8)
        ax.spines[:].set_color(cls.GRID_COLOR)
        ax.grid(True, color=cls.GRID_COLOR, linewidth=0.6, alpha=0.7)
        ax.set_ylim(0, 10.5)
        ax.set_ylabel("Score (0–10)", color="#888899", fontsize=8)
        ax.set_xlabel("Time (s)", color="#888899", fontsize=8)

    @classmethod
    def _add_avg_line(cls, ax, arr: List[float], color: str):
        if arr:
            avg = float(np.mean(arr))
            ax.axhline(avg, color=color, linewidth=1.0,
                       linestyle="--", alpha=0.6, label=f"Avg {avg:.1f}")

    @classmethod
    def _timeline_dashboard(cls, tl: TimelineTracker, agg: dict, out_dir: str) -> str:
        fig = plt.figure(figsize=(14, 8), facecolor=cls.DARK_BG)
        fig.suptitle("Session Overview — All Scores", color="#ffffff",
                     fontsize=13, fontweight="bold", y=0.97)

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35,
                               left=0.07, right=0.97, top=0.91, bottom=0.09)

        plots = [
            (0, 0, "Overall Score",    tl.overall,    "overall"),
            (0, 1, "Posture Score",    tl.posture,    "posture"),
            (0, 2, "Eye Contact",      tl.eye,        "eye"),
            (1, 0, "Movement Stability",tl.movement,  "movement"),
            (1, 1, "Engagement",       tl.engagement, "engagement"),
            (1, 2, "Attention",        tl.attention,  "attention"),
        ]

        for row, col, title, data, key in plots:
            ax = fig.add_subplot(gs[row, col])
            cls._setup_ax(ax, title)
            if tl.times and data:
                color = cls.LINE_COLORS[key]
                ax.plot(tl.times, data, color=color, linewidth=1.6,
                        alpha=0.9, label=title)
                # Smoothed trend line
                if len(data) > 8:
                    smooth = np.convolve(data, np.ones(8)/8, mode="valid")
                    t_trim = tl.times[7:][:len(smooth)]
                    ax.plot(t_trim, smooth, color=color, linewidth=2.5,
                            alpha=0.5, linestyle="-")
                cls._add_avg_line(ax, data, color)
                ax.legend(fontsize=7, loc="lower right",
                          facecolor=cls.DARK_BG, labelcolor="#ccccdd",
                          framealpha=0.6)

        path = os.path.join(out_dir, "01_overview_dashboard.png")
        fig.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor=cls.DARK_BG)
        plt.close(fig)
        return path

    @classmethod
    def _posture_detail(cls, tl: TimelineTracker, agg: dict, out_dir: str) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=cls.DARK_BG)
        fig.suptitle("Posture Analysis", color="#ffffff",
                     fontsize=12, fontweight="bold")

        ax1, ax2 = axes

        # Posture score timeline
        cls._setup_ax(ax1, "Posture Score Over Time")
        if tl.times and tl.posture:
            col = cls.LINE_COLORS["posture"]
            ax1.fill_between(tl.times, tl.posture, alpha=0.15, color=col)
            ax1.plot(tl.times, tl.posture, color=col, linewidth=1.8)
            cls._add_avg_line(ax1, tl.posture, col)
            # Threshold lines
            ax1.axhline(7.5, color="#00e87a", linewidth=0.8,
                        linestyle=":", alpha=0.5, label="Good threshold")
            ax1.axhline(5.0, color="#f5a623", linewidth=0.8,
                        linestyle=":", alpha=0.5, label="Moderate threshold")
            ax1.legend(fontsize=7, facecolor=cls.DARK_BG, labelcolor="#ccccdd")

        # Posture distribution pie
        ax2.set_facecolor(cls.DARK_BG)
        ax2.set_title("Posture Distribution", color="#ccccdd",
                      fontsize=10, pad=8, fontweight="bold")
        p = agg.get("percentages", {})
        straight = p.get("straight_posture_pct", 50)
        rest     = max(100 - straight, 0)
        wedge_colors = ["#00e87a", "#e05252"]
        wedges, texts, autotexts = ax2.pie(
            [straight, rest],
            labels=["Straight", "Non-Straight"],
            colors=wedge_colors, autopct="%1.1f%%",
            startangle=90, textprops={"color": "#ccccdd", "fontsize": 9})
        for at in autotexts:
            at.set_color("#ffffff")
            at.set_fontsize(9)

        for ax in axes:
            ax.spines[:].set_color(cls.GRID_COLOR)

        path = os.path.join(out_dir, "02_posture_detail.png")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=cls.DARK_BG)
        plt.close(fig)
        return path

    @classmethod
    def _gaze_movement(cls, tl: TimelineTracker, agg: dict, out_dir: str) -> str:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor=cls.DARK_BG,
                                 sharex=True)
        fig.suptitle("Eye Contact & Movement Analysis", color="#ffffff",
                     fontsize=12, fontweight="bold")

        ax1, ax2 = axes

        cls._setup_ax(ax1, "Eye Contact Score")
        if tl.times and tl.eye:
            col = cls.LINE_COLORS["eye"]
            ax1.fill_between(tl.times, tl.eye, alpha=0.12, color=col)
            ax1.plot(tl.times, tl.eye, color=col, linewidth=1.8, label="Eye Contact")
            cls._add_avg_line(ax1, tl.eye, col)
            ax1.legend(fontsize=8, facecolor=cls.DARK_BG, labelcolor="#ccccdd")

        cls._setup_ax(ax2, "Movement Stability Score")
        if tl.times and tl.movement:
            col = cls.LINE_COLORS["movement"]
            ax2.fill_between(tl.times, tl.movement, alpha=0.12, color=col)
            ax2.plot(tl.times, tl.movement, color=col, linewidth=1.8, label="Movement")
            cls._add_avg_line(ax2, tl.movement, col)
            ax2.legend(fontsize=8, facecolor=cls.DARK_BG, labelcolor="#ccccdd")
            ax2.set_xlabel("Time (s)", color="#888899", fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(out_dir, "03_gaze_movement.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=cls.DARK_BG)
        plt.close(fig)
        return path

    @classmethod
    def _engagement_attention(cls, tl: TimelineTracker, agg: dict, out_dir: str) -> str:
        fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=cls.DARK_BG)
        fig.suptitle("Engagement & Attention Timeline", color="#ffffff",
                     fontsize=12, fontweight="bold")

        cls._setup_ax(ax, "Engagement vs Attention")
        if tl.times:
            if tl.engagement:
                col = cls.LINE_COLORS["engagement"]
                ax.plot(tl.times, tl.engagement, color=col, linewidth=2.0,
                        label="Engagement", alpha=0.9)
                cls._add_avg_line(ax, tl.engagement, col)
            if tl.attention:
                col = cls.LINE_COLORS["attention"]
                ax.plot(tl.times, tl.attention, color=col, linewidth=2.0,
                        label="Attention", alpha=0.9, linestyle="--")
                cls._add_avg_line(ax, tl.attention, col)
            ax.legend(fontsize=9, facecolor=cls.DARK_BG, labelcolor="#ccccdd",
                      loc="lower right")

        path = os.path.join(out_dir, "04_engagement_attention.png")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=cls.DARK_BG)
        plt.close(fig)
        return path

    @classmethod
    def _event_timeline(cls, events: List[BehavioralEvent],
                        tl: TimelineTracker, out_dir: str) -> str:
        fig, ax = plt.subplots(figsize=(13, 3.5), facecolor=cls.DARK_BG)
        ax.set_facecolor(cls.DARK_BG)
        ax.set_title("Behavioral Events Timeline", color="#ccccdd",
                     fontsize=10, pad=8, fontweight="bold")
        ax.spines[:].set_color(cls.GRID_COLOR)
        ax.grid(True, color=cls.GRID_COLOR, linewidth=0.6, alpha=0.5, axis="x")

        if tl.overall:
            ax.plot(tl.times, tl.overall, color="#00e87a", linewidth=1.0,
                    alpha=0.3, label="Overall Score (background)")

        sev_colors = {"high": "#e05252", "medium": "#f5a623", "low": "#52d4e0"}
        if events:
            y_labels = []
            y_pos    = []
            for i, ev in enumerate(events):
                col = sev_colors.get(ev.severity, "#ccccdd")
                ax.axvline(ev.time, color=col, linewidth=1.2, alpha=0.8)
                ax.text(ev.time, 10.2 - (i % 3) * 0.6,
                        ev.event_type[:18], color=col, fontsize=6,
                        rotation=45, ha="left", va="bottom")

        ax.set_xlim(left=0)
        ax.set_ylim(0, 12)
        ax.set_xlabel("Time (s)", color="#888899", fontsize=8)
        ax.tick_params(colors="#888899", labelsize=8)
        # Legend for severities
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="#e05252", linewidth=2, label="High severity"),
            Line2D([0], [0], color="#f5a623", linewidth=2, label="Medium severity"),
            Line2D([0], [0], color="#52d4e0", linewidth=2, label="Low severity"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, facecolor=cls.DARK_BG,
                  labelcolor="#ccccdd", loc="upper right")

        path = os.path.join(out_dir, "05_event_timeline.png")
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=cls.DARK_BG)
        plt.close(fig)
        return path

    @classmethod
    def _scorecard_radar(cls, agg: dict, out_dir: str) -> str:
        """Generates a spider/radar chart of category scores."""
        avg  = agg.get("avg_scores", {})
        cats = ["Posture", "Eye Contact", "Movement", "Expression",
                "Gesture", "Engagement", "Attention"]
        vals = [
            avg.get("posture",    5),
            avg.get("eye_contact",5),
            avg.get("movement",   5),
            avg.get("expression", 5),
            avg.get("gesture",    5),
            avg.get("engagement", 5),
            avg.get("attention",  5),
        ]
        # Close the polygon
        vals += vals[:1]
        N     = len(cats)
        angles= [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), facecolor=cls.DARK_BG,
                               subplot_kw=dict(polar=True))
        ax.set_facecolor("#0d0d25")
        fig.suptitle("Performance Radar", color="#ffffff",
                     fontsize=12, fontweight="bold", y=0.98)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cats, color="#ccccdd", size=9)
        ax.set_rlabel_position(30)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(["2", "4", "6", "8", "10"],
                           color="#666677", size=7)
        ax.set_ylim(0, 10)
        ax.grid(color=cls.GRID_COLOR, linewidth=0.7)
        ax.spines["polar"].set_color(cls.GRID_COLOR)

        ax.plot(angles, vals, color="#00e87a", linewidth=2.0,
                linestyle="solid", alpha=0.9)
        ax.fill(angles, vals, color="#00e87a", alpha=0.18)

        # Reference circle at 7.5
        ref_vals = [7.5] * (N + 1)
        ax.plot(angles, ref_vals, color="#f5a623", linewidth=1.0,
                linestyle="--", alpha=0.5, label="Good threshold (7.5)")
        ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.15),
                  facecolor=cls.DARK_BG, labelcolor="#ccccdd")

        path = os.path.join(out_dir, "06_radar_scorecard.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=cls.DARK_BG)
        plt.close(fig)
        return path


# ══════════════════════════════════════════════════════════════════
#  INTERVIEW SCORECARD GENERATOR
# ══════════════════════════════════════════════════════════════════

class InterviewScorecard:
    """
    Generates a professional interview performance scorecard
    with grades, interpretations, strengths, and improvement areas.
    """

    GRADE_MAP = [
        (9.0, "A+", "Outstanding"),
        (8.0, "A",  "Excellent"),
        (7.0, "B+", "Very Good"),
        (6.0, "B",  "Good"),
        (5.0, "C",  "Moderate"),
        (4.0, "D",  "Below Average"),
        (0.0, "F",  "Needs Significant Improvement"),
    ]

    @classmethod
    def grade(cls, score: float) -> Tuple[str, str]:
        for threshold, letter, label in cls.GRADE_MAP:
            if score >= threshold:
                return letter, label
        return "F", "Needs Significant Improvement"

    @classmethod
    def interpret_posture(cls, score: float, pct_straight: float) -> str:
        if pct_straight >= 80:
            return (f"Maintained upright posture for {pct_straight:.0f}% of the session. "
                    "Strong postural discipline observed.")
        elif pct_straight >= 55:
            return (f"Maintained upright posture for {pct_straight:.0f}% of the session. "
                    "Moderate postural consistency with some deviation periods.")
        else:
            return (f"Upright posture was observed for {pct_straight:.0f}% of the session. "
                    "Significant postural variability noted — consider seated alignment awareness.")

    @classmethod
    def interpret_eye(cls, score: float, pct_camera: float) -> str:
        if pct_camera >= 70:
            return (f"Camera-directed gaze maintained {pct_camera:.0f}% of session. "
                    "Strong eye contact engagement signals observed.")
        elif pct_camera >= 45:
            return (f"Camera engagement at {pct_camera:.0f}% — moderate attention consistency. "
                    "Some distraction or gaze drift periods detected.")
        else:
            return (f"Camera-directed gaze at {pct_camera:.0f}% of session. "
                    "Low eye engagement signals — frequent gaze breaks noted.")

    @classmethod
    def interpret_movement(cls, score: float, pct_stable: float, pct_restless: float) -> str:
        if pct_stable >= 70:
            return (f"Body stability observed for {pct_stable:.0f}% of session. "
                    "Composed and controlled movement signals throughout.")
        elif pct_restless > 30:
            return (f"Elevated movement variance detected. Restless movement signals "
                    f"in {pct_restless:.0f}% of observations — may indicate physical discomfort.")
        else:
            return (f"Movement stability at {pct_stable:.0f}%. "
                    "Moderate movement variance with occasional activity spikes.")

    @classmethod
    def identify_strengths(cls, avg_scores: dict) -> List[str]:
        strengths = []
        label_map = {
            "posture":    "Posture & Physical Presence",
            "eye_contact":"Eye Contact & Camera Engagement",
            "movement":   "Movement Stability & Composure",
            "expression": "Facial Expression & Engagement Signals",
            "gesture":    "Hand Gesture Communication",
            "engagement": "Overall Behavioral Engagement",
            "attention":  "Attention & Focus Consistency",
        }
        for key, score in avg_scores.items():
            if score >= 7.5:
                strengths.append(f"{label_map.get(key, key)}: {score:.1f}/10")
        return strengths[:4]

    @classmethod
    def identify_improvements(cls, avg_scores: dict) -> List[str]:
        improvements = []
        advice_map = {
            "posture":     "Practice maintaining upright seated posture during mock interviews.",
            "eye_contact": "Focus on maintaining consistent camera gaze to simulate eye contact.",
            "movement":    "Minimize unnecessary body movement — practice stillness exercises.",
            "expression":  "Work on maintaining engaged and expressive facial signals.",
            "gesture":     "Use open-palm, deliberate hand gestures to reinforce communication.",
            "engagement":  "Practice active engagement techniques: nodding, leaning slightly forward.",
            "attention":   "Reduce environmental distractions; practice focused attention drills.",
        }
        for key, score in avg_scores.items():
            if score < 6.0:
                improvements.append(f"{key.replace('_', ' ').title()}: {advice_map.get(key, '')}")
        return improvements[:4]

    @classmethod
    def generate(cls, agg: dict, events: List[BehavioralEvent],
                 stats: dict) -> str:
        SEP  = "═" * 66
        sep2 = "─" * 66

        avg = agg.get("avg_scores",  {})
        p   = agg.get("percentages", {})
        t   = agg.get("trends",      {})
        dur = agg.get("duration_seconds", 0)

        overall    = avg.get("overall", 0)
        g_letter, g_label = cls.grade(overall)
        bp         = agg.get("behavioral_presence", overall)

        strengths   = cls.identify_strengths(avg)
        improvements= cls.identify_improvements(avg)

        def fmt_t(s: float) -> str:
            return f"{int(s)//60:02d}:{int(s)%60:02d}"

        lines = [
            SEP,
            "    ★  AI INTERVIEW BEHAVIORAL PERFORMANCE SCORECARD  ★",
            SEP,
            "",
            f"  Overall Behavioral Grade     : {g_letter}  —  {g_label}",
            f"  Overall Score                : {overall:.1f} / 10",
            f"  Behavioral Presence Score    : {bp:.1f} / 10",
            f"  Session Duration             : {fmt_t(dur)}",
            f"  Total Behavioral Events      : {agg.get('total_events', 0)}",
            "",
            sep2,
            "  CATEGORY SCORECARD",
            sep2,
        ]

        categories = [
            ("Posture",          "posture",     "B"),
            ("Eye Contact",      "eye_contact", "C"),
            ("Movement Stability","movement",   "D"),
            ("Facial Expression","expression",  "E"),
            ("Gesture Communication","gesture", "F"),
            ("Engagement",       "engagement",  "G"),
            ("Attention Focus",  "attention",   "H"),
        ]

        for label, key, ref in categories:
            sc = avg.get(key, 0)
            gl, gn = cls.grade(sc)
            bar = "█" * int(sc) + "░" * (10 - int(sc))
            trend = t.get(key.split("_")[0], "—")
            lines.append(
                f"  {label:<24} {sc:5.1f}/10  [{bar}]  {gl:3s} ({gn[:16]:<16})  ↑ {trend}")

        lines += [
            "",
            sep2,
            "  DETAILED INTERPRETATIONS",
            sep2,
            "",
            "  POSTURE:",
            f"    {cls.interpret_posture(avg.get('posture',0), p.get('straight_posture_pct',0))}",
            "",
            "  EYE CONTACT:",
            f"    {cls.interpret_eye(avg.get('eye_contact',0), p.get('at_camera_gaze_pct',0))}",
            "",
            "  MOVEMENT:",
            f"    {cls.interpret_movement(avg.get('movement',0), p.get('stable_movement_pct',0), p.get('restless_movement_pct',0))}",
            "",
        ]

        lines += [sep2, "  IDENTIFIED STRENGTHS", sep2]
        if strengths:
            for s in strengths:
                lines.append(f"  ✓  {s}")
        else:
            lines.append("  No standout strengths above threshold this session.")

        lines += ["", sep2, "  AREAS FOR IMPROVEMENT", sep2]
        if improvements:
            for imp in improvements:
                lines.append(f"  →  {imp}")
        else:
            lines.append("  Strong overall performance — continue current practice.")

        # Statistical highlights
        lines += ["", sep2, "  STATISTICAL HIGHLIGHTS", sep2]
        st = agg.get("statistics", {})
        for key in ["posture", "eye", "movement", "engagement"]:
            s = st.get(key, {})
            if s:
                lbl = key.replace("_", " ").title()
                lines.append(
                    f"  {lbl:<20}  avg={s.get('mean',0):.2f}  "
                    f"σ={s.get('std',0):.2f}  "
                    f"stability={s.get('stability_pct',0):.0f}%  "
                    f"volatility={s.get('volatility',0):.1f}%")

        lines += ["", sep2, "  BEHAVIORAL EVENTS SUMMARY", sep2]
        ec = agg.get("event_counts", {})
        if ec:
            for etype, cnt in sorted(ec.items(), key=lambda x: -x[1]):
                lines.append(f"  {etype:<35} ×{cnt}")
        else:
            lines.append("  No significant behavioral events recorded.")

        lines += ["", SEP, "  END OF SCORECARD", SEP]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#  REPORT GENERATOR  (fully upgraded)
# ══════════════════════════════════════════════════════════════════

class ReportGenerator:

    @staticmethod
    def _fmt(seconds: float) -> str:
        return f"{int(seconds)//60:02d}:{int(seconds)%60:02d}"

    @staticmethod
    def generate_txt(agg: dict, events: List[BehavioralEvent],
                     timeline: TimelineTracker, timestamp: str) -> str:
        SEP  = "═" * 66
        sep2 = "─" * 66
        p    = agg.get("percentages", {})
        avg  = agg.get("avg_scores",  {})
        t    = agg.get("trends",      {})
        dur  = agg.get("duration_seconds", 0)
        stats= agg.get("statistics",  {})

        def grade(score):
            if score >= 8.0:  return "Excellent"
            if score >= 6.5:  return "Good"
            if score >= 5.0:  return "Moderate"
            return "Needs Improvement"

        lines = [
            SEP,
            "    AI INTERVIEW BEHAVIOR ANALYZER — SESSION REPORT",
            f"    Generated : {timestamp}",
            f"    Duration  : {ReportGenerator._fmt(dur)}  ({dur:.0f}s)",
            SEP,
            "",
            "  ⚠  DISCLAIMER",
            "  This report documents observable behavioral signals only.",
            "  It does not assess personality, honesty, or psychological state.",
            "",
            sep2,
            "  A.  OVERALL SUMMARY",
            sep2,
            f"  Overall Behavioral Score     : {avg.get('overall',0):.1f} / 10  ({grade(avg.get('overall',0))})",
            f"  Behavioral Presence Score    : {agg.get('behavioral_presence',0):.1f} / 10",
            f"  Engagement Score             : {avg.get('engagement',0):.1f} / 10",
            f"  Attention Score              : {avg.get('attention',0):.1f} / 10",
            f"  Overall Session Trend        : {t.get('overall','N/A').upper()}",
            f"  Strongest Signal Area        : {agg.get('strongest_signal','N/A').title()}",
            f"  Weakest  Signal Area         : {agg.get('weakest_signal','N/A').title()}",
            f"  Total Behavioral Events      : {agg.get('total_events', 0)}",
            f"  Log Entries Captured         : {agg.get('total_log_entries',0)}",
            "",
            sep2,
            "  B.  POSTURE ANALYSIS",
            sep2,
            f"  Score                        : {avg.get('posture',0):.1f}/10  ({grade(avg.get('posture',0))})",
            f"  Upright Posture              : {p.get('straight_posture_pct',0):.1f}% of session",
            f"  Avg Composite Spine Angle    : {agg.get('avg_spine_angle',0):.1f}°",
            f"  Score Trend                  : {t.get('posture','N/A').upper()}",
        ]
        ps = stats.get("posture", {})
        if ps:
            lines += [
                f"  Score Std Deviation          : {ps.get('std',0):.2f}",
                f"  Score Volatility             : {ps.get('volatility',0):.1f}%",
                f"  Score Stability              : {ps.get('stability_pct',0):.0f}% within 1σ",
            ]
        lines += [
            f"  Interpretation               : Candidate displayed behavioral signals",
            f"    associated with upright posture for {p.get('straight_posture_pct',0):.1f}%",
            f"    of the session duration.",
            "",
            sep2,
            "  C.  EYE CONTACT ANALYSIS",
            sep2,
            f"  Score                        : {avg.get('eye_contact',0):.1f}/10  ({grade(avg.get('eye_contact',0))})",
            f"  Camera-Directed Gaze         : {p.get('at_camera_gaze_pct',0):.1f}% of session",
            f"  Downward Gaze Frequency      : {p.get('downward_gaze_pct',0):.1f}% of session",
            f"  Avg Gaze Attention Index     : {p.get('avg_gaze_attention',0)*100:.1f}%",
            f"  Score Trend                  : {t.get('eye','N/A').upper()}",
        ]
        es = stats.get("eye", {})
        if es:
            lines += [
                f"  Score Std Deviation          : {es.get('std',0):.2f}",
                f"  Stability                    : {es.get('stability_pct',0):.0f}%",
            ]
        lines += [
            f"  Interpretation               : Eye contact engagement signals were",
            f"    {'strong' if p.get('at_camera_gaze_pct',0)>60 else 'moderate' if p.get('at_camera_gaze_pct',0)>40 else 'limited'}"
            f" relative to total session time.",
            "",
            sep2,
            "  D.  EXPRESSION ANALYSIS",
            sep2,
            f"  Score                        : {avg.get('expression',0):.1f}/10  ({grade(avg.get('expression',0))})",
            f"  Smiling Detected             : {p.get('smiling_pct',0):.1f}% of session",
            f"  Neutral Expression           : {p.get('neutral_expression_pct',0):.1f}% of session",
            f"  Interpretation               : Candidate displayed primarily",
            f"    {'engaged (smiling)' if p.get('smiling_pct',0)>20 else 'composed (neutral)'}",
            f"    facial behavioral signals.",
            "",
            sep2,
            "  E.  GESTURE ANALYSIS",
            sep2,
            f"  Score                        : {avg.get('gesture',0):.1f}/10  ({grade(avg.get('gesture',0))})",
            f"  Open Palm Gestures           : {p.get('open_palm_gesture_pct',0):.1f}% of session",
            f"  Hands Not Visible            : {p.get('no_hands_visible_pct',0):.1f}% of session",
            f"  Interpretation               : Open-hand gesture signals are associated",
            f"    with communicative engagement indicators.",
            "",
            sep2,
            "  F.  MOVEMENT ANALYSIS",
            sep2,
            f"  Score                        : {avg.get('movement',0):.1f}/10  ({grade(avg.get('movement',0))})",
            f"  Stable Movement              : {p.get('stable_movement_pct',0):.1f}% of session",
            f"  Restless Movement            : {p.get('restless_movement_pct',0):.1f}% of session",
            f"  Score Trend                  : {t.get('movement','N/A').upper()}",
        ]
        ms = stats.get("movement", {})
        if ms:
            lines += [
                f"  Movement Volatility          : {ms.get('volatility',0):.1f}%",
                f"  Stability Score              : {ms.get('stability_pct',0):.0f}%",
            ]
        lines += [
            f"  Interpretation               : {'High movement stability signals observed.' if p.get('stable_movement_pct',0)>60 else 'Moderate or elevated movement variance detected.'}",
            "",
            sep2,
            "  G.  BEHAVIORAL EVENTS TIMELINE",
            sep2,
        ]

        if events:
            for ev in events:
                sev_tag = f"[{ev.severity.upper()}]"
                lines.append(f"  {ReportGenerator._fmt(ev.time)}  {sev_tag:<8} [{ev.event_type}]")
                lines.append(f"              {ev.detail}")
        else:
            lines.append("  No significant behavioral events detected.")

        lines += [
            "",
            sep2,
            "  H.  STATISTICAL SUMMARY",
            sep2,
            f"  {'Metric':<20} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7} {'Stability':>10}",
            f"  {'─'*20}  {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*10}",
        ]
        for key in ["posture", "eye", "movement", "expression", "gesture", "engagement", "attention"]:
            s = stats.get(key, {})
            if s:
                lbl = key.replace("_", " ").title()
                lines.append(
                    f"  {lbl:<20} {s.get('mean',0):>7.2f} {s.get('std',0):>7.2f} "
                    f"{s.get('min',0):>7.2f} {s.get('max',0):>7.2f} "
                    f"{s.get('stability_pct',0):>9.0f}%")

        lines += [
            "",
            sep2,
            "  I.  SCORE SUMMARY TABLE",
            sep2,
            f"  {'Category':<24} {'Avg Score':>10}   {'Grade':<20} {'Trend'}",
            f"  {'─'*24}  {'─'*10}   {'─'*20} {'─'*10}",
            f"  {'Posture':<24} {avg.get('posture',0):>9.1f}/10   {grade(avg.get('posture',0)):<20} {t.get('posture','N/A')}",
            f"  {'Movement':<24} {avg.get('movement',0):>9.1f}/10   {grade(avg.get('movement',0)):<20} {t.get('movement','N/A')}",
            f"  {'Eye Contact':<24} {avg.get('eye_contact',0):>9.1f}/10   {grade(avg.get('eye_contact',0)):<20} {t.get('eye','N/A')}",
            f"  {'Expression':<24} {avg.get('expression',0):>9.1f}/10   {grade(avg.get('expression',0)):<20} —",
            f"  {'Gesture':<24} {avg.get('gesture',0):>9.1f}/10   {grade(avg.get('gesture',0)):<20} —",
            f"  {'Engagement':<24} {avg.get('engagement',0):>9.1f}/10   {grade(avg.get('engagement',0)):<20} {t.get('engagement','N/A')}",
            f"  {'Attention':<24} {avg.get('attention',0):>9.1f}/10   {grade(avg.get('attention',0)):<20} {t.get('attention','N/A')}",
            f"  {'─'*24}  {'─'*10}   {'─'*20} {'─'*10}",
            f"  {'OVERALL':<24} {avg.get('overall',0):>9.1f}/10   {grade(avg.get('overall',0)):<20} {t.get('overall','N/A')}",
            "",
            SEP,
            "  END OF REPORT",
            SEP,
        ]
        return "\n".join(lines)

    @staticmethod
    def save(agg: dict, events: List[BehavioralEvent],
             timeline: TimelineTracker, logs: List[FrameLog]):
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_human = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create session subfolder
        session_dir  = os.path.join(REPORTS_DIR, f"session_{ts}")
        os.makedirs(session_dir, exist_ok=True)

        txt_path      = os.path.join(session_dir, "report.txt")
        json_path     = os.path.join(session_dir, "report.json")
        scorecard_path= os.path.join(session_dir, "scorecard.txt")

        # TXT report
        txt_content = ReportGenerator.generate_txt(agg, events, timeline, ts_human)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)

        # Scorecard
        stats = BehaviorStatistics.from_logs(logs)
        sc_content = InterviewScorecard.generate(agg, events, stats)
        with open(scorecard_path, "w", encoding="utf-8") as f:
            f.write(sc_content)

        # JSON
        json_payload = {
            "generated_at": ts_human,
            "aggregation":  agg,
            "events":       [e.to_dict() for e in events],
            "timeline":     timeline.to_dict(),
            "frame_logs":   [l.to_dict() for l in logs],
            "statistics":   stats,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2)

        # Graphs
        graph_paths = GraphGenerator.generate_all(timeline, agg, events, session_dir)

        _safe_log(f"\n[REPORT] Session dir -> {session_dir}")
        _safe_log(f"[REPORT] TXT       -> {txt_path}")
        _safe_log(f"[REPORT] Scorecard -> {scorecard_path}")
        _safe_log(f"[REPORT] JSON      -> {json_path}")
        if graph_paths:
            for gp in graph_paths:
                _safe_log(f"[GRAPH]  {gp}")
        try:
            _safe_log("\n" + sc_content)
        except Exception:
            pass
        return session_dir


# ══════════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════

def safe_lm_pose(landmarks, idx, w, h):
    lm = landmarks[idx]
    if lm.visibility < 0.4:
        return None
    return (lm.x * w, lm.y * h)


def face_pt(face_lms, idx, w, h):
    lm = face_lms.landmark[idx]
    return (lm.x * w, lm.y * h)


def dist(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) -
                                np.array(p2, dtype=np.float32)))


def angle_of_vector(p1, p2) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(np.degrees(np.arctan2(abs(dx), max(abs(dy), 1e-6))))


def exp_smooth(new_val: float, old_val: float,
               alpha: float = EXP_SMOOTH_ALPHA) -> float:
    if old_val is None:
        return new_val
    return alpha * new_val + (1.0 - alpha) * old_val


def fmt_time(seconds: float) -> str:
    return f"{int(seconds)//60:02d}:{int(seconds)%60:02d}"


# ══════════════════════════════════════════════════════════════════
#  MODULE 1 — POSTURE DETECTION  (upgraded with calibration)
# ══════════════════════════════════════════════════════════════════

def detect_posture(landmarks, w, h, calibrator: PostureCalibrator):
    l_sh = safe_lm_pose(landmarks, LM.LEFT_SHOULDER,  w, h)
    r_sh = safe_lm_pose(landmarks, LM.RIGHT_SHOULDER, w, h)
    l_hi = safe_lm_pose(landmarks, LM.LEFT_HIP,       w, h)
    r_hi = safe_lm_pose(landmarks, LM.RIGHT_HIP,      w, h)
    l_ea = safe_lm_pose(landmarks, LM.LEFT_EAR,       w, h)
    r_ea = safe_lm_pose(landmarks, LM.RIGHT_EAR,      w, h)

    if not all([l_sh, r_sh, l_hi, r_hi]):
        return "Unknown", 0.0, 0.0, 0.0

    sh_mid      = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    hi_mid      = ((l_hi[0] + r_hi[0]) / 2, (l_hi[1] + r_hi[1]) / 2)
    spine_angle = angle_of_vector(hi_mid, sh_mid)

    # Shoulder roll (asymmetry %)
    shoulder_roll = abs(l_sh[1] - r_sh[1]) / max(w, 1) * 100

    neck_angle = 0.0
    if l_ea and r_ea:
        ear_mid    = ((l_ea[0] + r_ea[0]) / 2, (l_ea[1] + r_ea[1]) / 2)
        neck_angle = angle_of_vector(sh_mid, ear_mid)

    # Forward lean: if shoulders significantly above expected hip-to-shoulder midpoint
    lean_score = 0.0
    if sh_mid[1] < hi_mid[1]:  # shoulders above hips (normal)
        torso_h = abs(hi_mid[1] - sh_mid[1])
        if torso_h > 0:
            # Horizontal displacement of shoulder vs hip midpoints
            h_disp = abs(sh_mid[0] - hi_mid[0]) / max(torso_h, 1) * 10
            lean_score = h_disp

    composite = (spine_angle * 0.50 + neck_angle * 0.28 +
                 shoulder_roll * 0.12 + lean_score * 0.10)

    # Feed calibrator and get adjusted composite
    calibrator.feed(composite)
    adj_composite = calibrator.adjust(composite)

    if adj_composite < SPINE_STRAIGHT_MAX:
        status = "Straight"
    elif adj_composite < SPINE_SLIGHT_MAX:
        status = "Slightly Slouched"
    elif adj_composite < SPINE_SLOUCH_MAX:
        status = "Slouched"
    else:
        status = "Heavy Slouch"

    # Leaning detection (raw composite)
    if composite > 35 and shoulder_roll > 3.5:
        # Determine lean direction from shoulder height asymmetry
        if l_sh[1] < r_sh[1]:
            status = "Leaning Left"
        else:
            status = "Leaning Right"

    return status, composite, spine_angle, neck_angle


# ══════════════════════════════════════════════════════════════════
#  MODULE 2 — HEAD TILT (FACE MESH)
# ══════════════════════════════════════════════════════════════════

def detect_head_tilt_face(face_lms, w, h):
    nose    = face_pt(face_lms, NOSE_TIP,    w, h)
    chin    = face_pt(face_lms, CHIN,        w, h)
    l_cheek = face_pt(face_lms, LEFT_CHEEK,  w, h)
    r_cheek = face_pt(face_lms, RIGHT_CHEEK, w, h)

    face_width  = max(dist(l_cheek, r_cheek), 1)
    face_height = max(dist(nose, chin), 1)
    dx_roll     = r_cheek[1] - l_cheek[1]
    roll_deg    = float(np.degrees(np.arctan2(dx_roll, face_width)))

    if roll_deg > 9:
        lateral = "Tilted Right"
    elif roll_deg < -9:
        lateral = "Tilted Left"
    else:
        lateral = "Center"

    nose_x_norm = (nose[0] - l_cheek[0]) / face_width
    if nose_x_norm < 0.33:
        forward = "Turned Right"
    elif nose_x_norm > 0.67:
        forward = "Turned Left"
    else:
        nose_chin_x   = abs(chin[0] - nose[0])
        forward_ratio = nose_chin_x / face_height
        if forward_ratio > 0.20:
            forward = "Forward"
        else:
            forward = "Neutral"

    return lateral, forward, abs(roll_deg)


# ══════════════════════════════════════════════════════════════════
#  MODULE 3 — EYE CONTACT / GAZE  (significantly upgraded)
# ══════════════════════════════════════════════════════════════════

def detect_gaze(face_lms, w, h, gaze_buffer: GazeBuffer):
    """
    Returns:
      status (str), gaze_ratio (float), vert_ratio (float),
      attention_index (0-1), eye_ar (float)
    """
    try:
        # ── Iris centers ──
        l_iris_pts = [face_pt(face_lms, i, w, h) for i in LEFT_IRIS]
        r_iris_pts = [face_pt(face_lms, i, w, h) for i in RIGHT_IRIS]
        l_iris_cx  = float(np.mean([p[0] for p in l_iris_pts]))
        l_iris_cy  = float(np.mean([p[1] for p in l_iris_pts]))
        r_iris_cx  = float(np.mean([p[0] for p in r_iris_pts]))
        r_iris_cy  = float(np.mean([p[1] for p in r_iris_pts]))

        # ── Eye corner coordinates ──
        l_eye_l = face_pt(face_lms, LEFT_EYE_LEFT,   w, h)
        l_eye_r = face_pt(face_lms, LEFT_EYE_RIGHT,  w, h)
        r_eye_l = face_pt(face_lms, RIGHT_EYE_LEFT,  w, h)
        r_eye_r = face_pt(face_lms, RIGHT_EYE_RIGHT, w, h)
        l_eye_w = max(dist(l_eye_l, l_eye_r), 1)
        r_eye_w = max(dist(r_eye_l, r_eye_r), 1)

        # Horizontal gaze ratio (0=far left, 1=far right)
        l_ratio    = (l_iris_cx - l_eye_l[0]) / l_eye_w
        r_ratio    = (r_iris_cx - r_eye_l[0]) / r_eye_w
        raw_h      = (l_ratio + r_ratio) / 2

        # ── Vertical gaze ──
        l_eye_top = face_pt(face_lms, LEFT_EYE_TOP,  w, h)
        l_eye_bot = face_pt(face_lms, LEFT_EYE_BOT,  w, h)
        r_eye_top = face_pt(face_lms, RIGHT_EYE_TOP, w, h)
        r_eye_bot = face_pt(face_lms, RIGHT_EYE_BOT, w, h)
        l_eye_h   = max(dist(l_eye_top, l_eye_bot), 1)
        r_eye_h   = max(dist(r_eye_top, r_eye_bot), 1)

        l_vert    = (l_iris_cy - l_eye_top[1]) / l_eye_h
        r_vert    = (r_iris_cy - r_eye_top[1]) / r_eye_h
        raw_v     = (l_vert + r_vert) / 2

        # Eye Aspect Ratio (EAR) for closure detection
        l_ear_val = l_eye_h / l_eye_w
        r_ear_val = r_eye_h / r_eye_w
        eye_ar    = (l_ear_val + r_ear_val) / 2

        # ── Smooth gaze ratios ──
        gaze_ratio, vert_ratio = gaze_buffer.update(raw_h, raw_v)

        # ── Classify gaze direction ──
        at_camera = (GAZE_H_INNER <= gaze_ratio <= GAZE_H_OUTER and
                     GAZE_V_INNER <= vert_ratio <= GAZE_V_OUTER)

        if eye_ar < EYE_CLOSED_RATIO:
            status = "Eyes Closed"
        elif at_camera:
            status = "At Camera"
        elif gaze_ratio < GAZE_H_INNER:
            status = "Looking Left"
        elif gaze_ratio > GAZE_H_OUTER:
            status = "Looking Right"
        elif vert_ratio < GAZE_V_INNER:
            status = "Looking Up"
        else:
            status = "Looking Down"

        # Attention index: how centered is gaze (0=far away, 1=perfect center)
        h_dev    = abs(gaze_ratio - 0.5) * 2   # 0–1
        v_dev    = abs(vert_ratio - 0.5) * 2
        att_idx  = max(1.0 - (h_dev * 0.6 + v_dev * 0.4), 0.0)

        return status, gaze_ratio, vert_ratio, att_idx, eye_ar

    except Exception:
        return "Unknown", 0.5, 0.5, 0.5, 0.25


# ══════════════════════════════════════════════════════════════════
#  MODULE 4 — EXPRESSION  (upgraded)
# ══════════════════════════════════════════════════════════════════

def detect_expression(face_lms, w, h):
    """
    Returns expression label, smile_ratio, eye_ar, brow_tension_index.
    Brow tension is a stress-related facial signal approximation only.
    """
    try:
        mouth_l = face_pt(face_lms, MOUTH_LEFT,   w, h)
        mouth_r = face_pt(face_lms, MOUTH_RIGHT,  w, h)
        mouth_t = face_pt(face_lms, MOUTH_TOP,    w, h)
        mouth_b = face_pt(face_lms, MOUTH_BOTTOM, w, h)

        mouth_width  = dist(mouth_l, mouth_r)
        mouth_height = max(dist(mouth_t, mouth_b), 1)
        smile_ratio  = mouth_width / mouth_height

        l_top = face_pt(face_lms, LEFT_EYE_TOP,   w, h)
        l_bot = face_pt(face_lms, LEFT_EYE_BOT,   w, h)
        r_top = face_pt(face_lms, RIGHT_EYE_TOP,  w, h)
        r_bot = face_pt(face_lms, RIGHT_EYE_BOT,  w, h)
        l_ew  = face_pt(face_lms, LEFT_EYE_LEFT,  w, h)
        l_er  = face_pt(face_lms, LEFT_EYE_RIGHT, w, h)
        r_ew  = face_pt(face_lms, RIGHT_EYE_LEFT, w, h)
        r_er  = face_pt(face_lms, RIGHT_EYE_RIGHT,w, h)

        l_eye_w = max(dist(l_ew, l_er), 1)
        r_eye_w = max(dist(r_ew, r_er), 1)
        l_ear   = dist(l_top, l_bot) / l_eye_w
        r_ear   = dist(r_top, r_bot) / r_eye_w
        avg_ear = (l_ear + r_ear) / 2

        # Brow tension approximation (inner brow distance vs face width)
        try:
            l_brow = face_pt(face_lms, LEFT_BROW_INNER,  w, h)
            r_brow = face_pt(face_lms, RIGHT_BROW_INNER, w, h)
            nose   = face_pt(face_lms, NOSE_TIP, w, h)
            l_ch   = face_pt(face_lms, LEFT_CHEEK,  w, h)
            r_ch   = face_pt(face_lms, RIGHT_CHEEK, w, h)
            face_w = max(dist(l_ch, r_ch), 1)
            brow_sep  = dist(l_brow, r_brow)
            # Low separation = brows pulled together = tension indicator
            brow_tension = max(1.0 - brow_sep / (face_w * 0.35), 0.0)
        except Exception:
            brow_tension = 0.0

        blinking = avg_ear < BLINK_THRESHOLD

        # Filtered smile: require meaningful smile_ratio AND no blinking
        if smile_ratio > SMILE_RATIO_THRESHOLD and not blinking and avg_ear > 0.18:
            return "Smiling", smile_ratio, avg_ear, brow_tension
        elif avg_ear < EYE_CLOSED_RATIO:
            return "Eyes Closed", smile_ratio, avg_ear, brow_tension
        elif blinking:
            return "Blinking", smile_ratio, avg_ear, brow_tension
        elif brow_tension > 0.55:
            return "Tense", smile_ratio, avg_ear, brow_tension

        return "Neutral", smile_ratio, avg_ear, brow_tension

    except Exception:
        return "Unknown", 3.0, 0.25, 0.0


# ══════════════════════════════════════════════════════════════════
#  MODULE 5 — HAND GESTURE  (upgraded)
# ══════════════════════════════════════════════════════════════════

FINGERTIPS  = [4, 8, 12, 16, 20]
FINGER_MIDS = [3, 6, 10, 14, 18]
WRIST_IDX   = 0


def classify_hand(hand_lms, w, h) -> str:
    wrist    = (hand_lms.landmark[WRIST_IDX].x * w,
                hand_lms.landmark[WRIST_IDX].y * h)
    extended = 0
    for tip, mid in zip(FINGERTIPS[1:], FINGER_MIDS[1:]):
        tip_pt = (hand_lms.landmark[tip].x * w, hand_lms.landmark[tip].y * h)
        mid_pt = (hand_lms.landmark[mid].x * w, hand_lms.landmark[mid].y * h)
        if dist(tip_pt, wrist) > dist(mid_pt, wrist) * 1.05:
            extended += 1
    thumb_tip = (hand_lms.landmark[4].x * w, hand_lms.landmark[4].y * h)
    thumb_ip  = (hand_lms.landmark[3].x * w, hand_lms.landmark[3].y * h)
    if dist(thumb_tip, wrist) > dist(thumb_ip, wrist):
        extended += 1
    if extended >= 4:
        return "Open Palm"
    elif extended == 0:
        return "Closed Fist"
    elif extended == 1:
        return "Pointing"
    return "Partial"


def detect_gestures(hand_results, w, h):
    if not hand_results or not hand_results.multi_hand_landmarks:
        return "No Hands", 0, []
    gestures = [classify_hand(lm, w, h) for lm in hand_results.multi_hand_landmarks]
    combined = f"{gestures[0]} / {gestures[1]}" if len(gestures) == 2 else gestures[0]
    return combined, len(gestures), gestures


# ══════════════════════════════════════════════════════════════════
#  MODULE 6 — MOVEMENT DETECTION  (upgraded)
# ══════════════════════════════════════════════════════════════════

TRACKED_LMS = [LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
               LM.LEFT_WRIST, LM.RIGHT_WRIST]
VECTOR_LEN  = len(TRACKED_LMS) * 2


def detect_movement(landmarks, w, h, history: deque, last_valid: list):
    coords = []
    for i, idx in enumerate(TRACKED_LMS):
        pt   = safe_lm_pose(landmarks, idx, w, h)
        base = i * 2
        if pt:
            last_valid[base]     = exp_smooth(pt[0] / w, last_valid[base])
            last_valid[base + 1] = exp_smooth(pt[1] / h, last_valid[base + 1])
        coords.append(last_valid[base])
        coords.append(last_valid[base + 1])

    if len(coords) != VECTOR_LEN:
        coords = (coords + [0.0] * VECTOR_LEN)[:VECTOR_LEN]

    history.append(coords)
    if len(history) < 5:
        return "Calibrating", 0.0

    try:
        arr      = np.array(list(history), dtype=np.float32)
        variance = float(np.mean(np.var(arr, axis=0)))

        # Jitter reduction: compare smoothed vs raw velocity
        if len(history) >= 3:
            last3    = np.array(list(history)[-3:], dtype=np.float32)
            velocity = float(np.mean(np.abs(np.diff(last3, axis=0))))
            # Blend variance and velocity for robustness
            combined = variance * 0.65 + velocity * 0.35
        else:
            combined = variance

    except Exception:
        return "Error", 0.0

    if combined < MOVEMENT_STABLE_MAX:
        return "Stable", variance
    elif combined < MOVEMENT_MODERATE_MAX:
        return "Moderate", variance
    elif combined < MOVEMENT_RESTLESS_MIN:
        return "Active", variance
    return "Restless", variance


# ══════════════════════════════════════════════════════════════════
#  SCORING ENGINE  (upgraded — engagement + attention scoring)
# ══════════════════════════════════════════════════════════════════

def calculate_scores(posture_status: str, movement_status: str, gaze_status: str,
                     expression: str, gesture: str,
                     composite_angle: float, variance: float,
                     gaze_attention: float, eye_ar: float):
    """
    Returns:
      posture_score, movement_score, eye_score, expression_score,
      gesture_score, engagement_score, attention_score
    All scores 0–10.
    """
    # ── Posture ──
    if posture_status == "Straight":
        posture_score = max(10 - composite_angle * 0.20, 7.5)
    elif posture_status == "Slightly Slouched":
        posture_score = max(10 - composite_angle * 0.40, 4.0)
    elif posture_status == "Slouched":
        posture_score = max(10 - composite_angle * 0.55, 2.0)
    elif posture_status == "Heavy Slouch":
        posture_score = max(10 - composite_angle * 0.70, 0.5)
    elif posture_status in ("Leaning Left", "Leaning Right"):
        posture_score = 4.5
    else:
        posture_score = 5.0

    # ── Movement ──
    if movement_status == "Stable":
        movement_score = 10.0
    elif movement_status == "Moderate":
        movement_score = max(10 - variance * 650, 6.0)
    elif movement_status == "Active":
        movement_score = max(10 - variance * 900, 4.0)
    elif movement_status == "Restless":
        movement_score = max(10 - variance * 1200, 0.0)
    else:
        movement_score = 5.0

    # ── Eye contact ──
    eye_map = {
        "At Camera":    10.0,
        "Looking Left":  3.5,
        "Looking Right": 3.5,
        "Looking Up":    4.5,
        "Looking Down":  2.5,
        "Eyes Closed":   1.0,
        "Unknown":       5.0,
    }
    eye_score = eye_map.get(gaze_status, 5.0)

    # ── Expression ──
    expr_map = {
        "Smiling":     10.0,
        "Neutral":      7.5,
        "Blinking":     5.0,
        "Tense":        4.0,
        "Eyes Closed":  2.0,
        "Unknown":      5.0,
    }
    expression_score = expr_map.get(expression, 5.0)

    # ── Gesture ──
    gest_map = {
        "No Hands":   6.0,
        "Open Palm":  10.0,
        "Pointing":    8.0,
        "Partial":     7.0,
        "Closed Fist": 4.0,
    }
    base_gest   = gesture.split(" / ")[0].strip()
    gesture_score = gest_map.get(base_gest, 5.0)

    # ── Engagement score (composite behavioral signal) ──
    # Engagement reflects active, positive participation signals
    engagement_score = (
        eye_score        * 0.35 +
        expression_score * 0.25 +
        posture_score    * 0.20 +
        gesture_score    * 0.10 +
        movement_score   * 0.10
    )

    # ── Attention score (focus/concentration index) ──
    # Attention reflects sustained, stable, camera-directed focus
    attention_score = (
        gaze_attention * 10  * 0.50 +
        eye_score            * 0.30 +
        movement_score       * 0.20
    )

    return (
        round(min(posture_score,     10.0), 1),
        round(min(movement_score,    10.0), 1),
        round(min(eye_score,         10.0), 1),
        round(min(expression_score,  10.0), 1),
        round(min(gesture_score,     10.0), 1),
        round(min(engagement_score,  10.0), 1),
        round(min(attention_score,   10.0), 1),
    )


# ══════════════════════════════════════════════════════════════════
#  OVERLAY RENDERING  (upgraded with meters and warning panel)
# ══════════════════════════════════════════════════════════════════

def _score_color(score: float):
    if score >= 7.5:
        return COLOR_GREEN
    elif score >= 5.0:
        return COLOR_YELLOW
    return COLOR_RED


def _status_color(status: str):
    good = {"Straight", "Center", "Neutral", "Stable", "At Camera",
            "Smiling", "Open Palm", "Pointing"}
    warn = {"Slightly Slouched", "Moderate", "Active", "Calibrating",
            "Partial", "Blinking", "No Hands", "Forward", "Backward", "Tense"}
    if status in good:
        return COLOR_GREEN
    if status in warn:
        return COLOR_YELLOW
    return COLOR_RED


def _warn_color(warn_type: Optional[str]) -> tuple:
    """Return warning color based on severity."""
    if not warn_type:
        return COLOR_WARN
    high_warns = {"Eyes Closed", "Face Not Visible", "Heavy Slouch",
                  "Restless Behavior", "Nervous Movement"}
    if warn_type in high_warns:
        return (0, 60, 200)   # bright red (BGR)
    return (0, 100, 255)      # orange-red


def draw_meter(frame, x: int, y: int, width: int, height: int,
               value: float, max_val: float, color, label: str):
    """Draw a horizontal meter bar with label."""
    ratio  = min(value / max(max_val, 0.01), 1.0)
    filled = int(width * ratio)
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (35, 35, 52), -1)
    # Filled portion
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + height), color, -1)
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (70, 70, 90), 1)
    # Label
    cv2.putText(frame, label, (x, y - 3), FONT, 0.35, COLOR_MUTED, 1, cv2.LINE_AA)


def draw_overlay(frame, data: dict, fps: float,
                 session_active: bool, elapsed: float,
                 remaining: float, overall_score: float,
                 warning: Optional[str], event_count: int,
                 calibrator: PostureCalibrator,
                 distraction_count: int = 0):

    h, w  = frame.shape[:2]
    PANEL = 308

    # Semi-transparent panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (PANEL, h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.line(frame, (PANEL, 0), (PANEL, h), COLOR_DIVIDER, 1)

    # Header
    cv2.rectangle(frame, (0, 0), (PANEL, 46), COLOR_HEADER, -1)
    cv2.putText(frame, "INTERVIEW ANALYZER v4", (8, 30),
                FONT, 0.54, COLOR_ACCENT, 2, cv2.LINE_AA)

    fps_col = COLOR_GREEN if fps >= 20 else COLOR_YELLOW if fps >= 12 else COLOR_RED
    cv2.putText(frame, f"{fps:.0f}fps", (w - 95, 28),
                FONT, 0.56, fps_col, 2, cv2.LINE_AA)

    y = 54

    def section(title: str):
        nonlocal y
        cv2.putText(frame, title, (8, y), FONT, 0.38, COLOR_MUTED, 1, cv2.LINE_AA)
        y += 4
        cv2.line(frame, (8, y), (PANEL - 8, y), COLOR_DIVIDER, 1)
        y += 12

    def row(label: str, value: str, color=COLOR_WHITE):
        nonlocal y
        cv2.putText(frame, label,      (10, y),      FONT, 0.38, COLOR_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, str(value), (10, y + 15), FONT, 0.52, color,       2, cv2.LINE_AA)
        y += 30

    def score_bar(lbl: str, score: float):
        nonlocal y
        bar_w  = PANEL - 20
        filled = int(bar_w * score / 10)
        col    = _score_color(score)
        cv2.putText(frame, f"{lbl}  {score:.1f}", (10, y),
                    FONT, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)
        y += 5
        cv2.rectangle(frame, (10, y), (10 + bar_w, y + 7), (38, 38, 55), -1)
        if filled > 0:
            cv2.rectangle(frame, (10, y), (10 + filled, y + 7), col, -1)
        y += 15

    # ── Session block ──
    section("SESSION")
    if session_active:
        s_txt = "● RECORDING"
        s_col = COLOR_GREEN
    else:
        s_txt = "○ IDLE  [S] to Start"
        s_col = COLOR_YELLOW

    # Calibration indicator
    if session_active and not calibrator.calibrated:
        prog_pct = int(calibrator.progress * 100)
        s_txt    = f"◉ CALIBRATING {prog_pct}%"
        s_col    = COLOR_CYAN

    cv2.putText(frame, s_txt, (10, y), FONT, 0.48, s_col, 2, cv2.LINE_AA)
    y += 17
    remain_str = fmt_time(remaining) if INTERVIEW_DURATION > 0 else "∞"
    cv2.putText(frame,
                f"Elapsed {fmt_time(elapsed)}   Left {remain_str}",
                (10, y), FONT, 0.40, COLOR_TIMER, 1, cv2.LINE_AA)
    y += 14
    cv2.putText(frame, f"Events: {event_count}  Distractions: {distraction_count}",
                (10, y), FONT, 0.38, COLOR_MUTED, 1, cv2.LINE_AA)
    y += 20

    # ── Overall score ──
    section("OVERALL SCORE")
    oc     = _score_color(overall_score)
    bar_w  = PANEL - 20
    filled = int(bar_w * overall_score / 10)
    cv2.putText(frame, f"{overall_score:.1f} / 10", (10, y),
                FONT, 0.68, oc, 2, cv2.LINE_AA)
    y += 8
    cv2.rectangle(frame, (10, y), (10 + bar_w, y + 9), (38, 38, 55), -1)
    if filled > 0:
        cv2.rectangle(frame, (10, y), (10 + filled, y + 9), oc, -1)
    y += 18

    # ── Engagement / Attention meters ──
    section("LIVE METERS")
    eng  = data.get("s_engagement",  5.0)
    att  = data.get("s_attention",   5.0)
    draw_meter(frame, 10, y + 3, PANEL - 22, 7,
               eng, 10.0, COLOR_CYAN, "Engagement")
    y += 20
    draw_meter(frame, 10, y + 3, PANEL - 22, 7,
               att, 10.0, COLOR_ORANGE, "Attention")
    y += 20
    draw_meter(frame, 10, y + 3, PANEL - 22, 7,
               data.get("s_movement", 5.0), 10.0, COLOR_PURPLE, "Stability")
    y += 22

    # ── Warning panel ──
    if warning:
        warn_col = _warn_color(warning)
        cv2.rectangle(frame, (0, y - 2), (PANEL, y + 20), (0, 25, 70), -1)
        cv2.rectangle(frame, (0, y - 2), (PANEL, y + 20), warn_col, 1)
        cv2.putText(frame, f"⚠ {warning[:28]}", (7, y + 14),
                    FONT, 0.42, warn_col, 1, cv2.LINE_AA)
    y += 24

    # ── Live signals ──
    section("POSTURE")
    row("Status", data["posture_status"], _status_color(data["posture_status"]))
    y -= 4

    section("HEAD / GAZE")
    row("Head", data["lateral"],  _status_color(data["lateral"]))
    row("Gaze", data["gaze"],     _status_color(data["gaze"]))
    y -= 4

    section("EXPRESSION / GESTURE")
    row("Expr",    data["expression"], _status_color(data["expression"]))
    row("Gesture", data["gesture"],    _status_color(data["gesture"].split("/")[0].strip()))
    y -= 4

    section("MOVEMENT")
    row("Status", data["movement"], _status_color(data["movement"]))
    y += 2

    section("SCORES")
    score_bar("Posture  ", data["s_posture"])
    score_bar("Movement ", data["s_movement"])
    score_bar("Eye Cntct", data["s_eye"])
    score_bar("Expressn ", data["s_expression"])
    score_bar("Gesture  ", data["s_gesture"])

    hint_y = min(y + 4, h - 10)
    cv2.putText(frame, "[S] Start  [E] End+Report  [Q] Quit",
                (8, hint_y), FONT, 0.35, COLOR_MUTED, 1, cv2.LINE_AA)

    # ── Session quality indicator (top-right corner) ──
    q_labels = [(8.5, "EXCELLENT", COLOR_GREEN), (7.0, "GOOD", COLOR_GREEN),
                (5.5, "MODERATE",  COLOR_YELLOW), (0.0, "LOW",  COLOR_RED)]
    q_label, q_col = "LOW", COLOR_RED
    for threshold, lbl, col in q_labels:
        if overall_score >= threshold:
            q_label, q_col = lbl, col
            break
    cv2.putText(frame, f"Quality: {q_label}", (PANEL + 10, 30),
                FONT, 0.50, q_col, 2, cv2.LINE_AA)

    return frame


# ══════════════════════════════════════════════════════════════════
#  SESSION LIFECYCLE  (Phase 3A — reusable start / end control)
# ══════════════════════════════════════════════════════════════════

@dataclass
class SessionLifecycle:
    session_active:  bool              = False
    session_start_t: float             = 0.0
    elapsed:         float             = 0.0
    session_logger:  SessionLogger       = field(default_factory=SessionLogger)
    timeline_tracker: TimelineTracker  = field(default_factory=TimelineTracker)
    event_detector:  EventDetector     = field(default_factory=EventDetector)
    calibrator:      PostureCalibrator = field(default_factory=PostureCalibrator)
    gaze_ratio_buf:  GazeBuffer        = field(default_factory=GazeBuffer)
    loop_active_logged: bool           = False
    stop_in_progress: bool             = False
    boot_ready:      bool              = False

    @property
    def lifecycle_id(self) -> int:
        return id(self)


def start_session(lifecycle: SessionLifecycle) -> bool:
    """Start a new analysis session. Returns True if a session was started."""
    if lifecycle.session_active:
        _safe_log(f"[SESSION] start skipped {{ session_active: true, lifecycle_id: {lifecycle.lifecycle_id} }}")
        return False
    lifecycle.session_active      = True
    lifecycle.session_start_t     = time.time()
    lifecycle.elapsed             = 0.0
    lifecycle.loop_active_logged  = False
    lifecycle.stop_in_progress    = False
    lifecycle.session_logger      = SessionLogger()
    lifecycle.timeline_tracker    = TimelineTracker()
    lifecycle.event_detector      = EventDetector()
    lifecycle.calibrator          = PostureCalibrator()
    lifecycle.gaze_ratio_buf      = GazeBuffer()
    _safe_log(
        f"[SESSION] started {{ session_active: true, start_t: {lifecycle.session_start_t}, "
        f"lifecycle_id: {lifecycle.lifecycle_id} }}"
    )
    return True


def end_session(lifecycle: SessionLifecycle) -> dict:
    """End the active session, aggregate results, and save reports."""
    lifecycle_id = lifecycle.lifecycle_id
    active = lifecycle.session_active
    elapsed = lifecycle.elapsed
    _safe_log(
        f"[SESSION] stop requested {{ active: {active}, elapsed: {elapsed:.2f}, "
        f"lifecycle_id: {lifecycle_id} }}"
    )

    if not lifecycle.session_active:
        return {
            "ok": False,
            "reason": "no_active_session",
            "sessionActive": False,
            "elapsed": round(elapsed, 2),
            "lifecycleId": lifecycle_id,
        }

    if lifecycle.elapsed <= 2.0:
        lifecycle.session_active = False
        _safe_log(
            f"[SESSION] stop skipped — elapsed too short ({elapsed:.2f}s <= 2.0s, "
            f"lifecycle_id: {lifecycle_id})"
        )
        return {
            "ok": False,
            "reason": "elapsed_too_short",
            "sessionActive": False,
            "elapsed": round(elapsed, 2),
            "lifecycleId": lifecycle_id,
        }

    lifecycle.stop_in_progress = True
    lifecycle.session_active = False
    _safe_log(f"[SESSION] report generation started {{ lifecycle_id: {lifecycle_id} }}")

    try:
        agg = SessionAggregator.compute(
            lifecycle.session_logger.get_logs(), lifecycle.timeline_tracker,
            lifecycle.elapsed, lifecycle.event_detector.get_events())
        session_dir = ReportGenerator.save(
            agg, lifecycle.event_detector.get_events(),
            lifecycle.timeline_tracker, lifecycle.session_logger.get_logs())
        _safe_log(
            f"[SESSION] report generation completed {{ success: true, session_dir: {session_dir}, "
            f"lifecycle_id: {lifecycle_id} }}"
        )
        return {
            "ok": True,
            "reason": "report_saved",
            "sessionActive": False,
            "elapsed": round(elapsed, 2),
            "sessionDir": session_dir,
            "lifecycleId": lifecycle_id,
        }
    except Exception as error:
        _safe_log(
            f"[SESSION] report generation failed {{ error: {error}, lifecycle_id: {lifecycle_id} }}"
        )
        return {
            "ok": False,
            "reason": "report_failed",
            "sessionActive": False,
            "elapsed": round(elapsed, 2),
            "error": str(error),
            "lifecycleId": lifecycle_id,
        }
    finally:
        lifecycle.stop_in_progress = False


# ══════════════════════════════════════════════════════════════════
#  ANALYZER CONTROLLER  (Phase 3B — external control layer)
# ══════════════════════════════════════════════════════════════════

class AnalyzerController:
    """Programmatic control surface for analyzer session lifecycle."""

    def __init__(self, lifecycle: SessionLifecycle):
        self.lifecycle = lifecycle
        self._shutdown_requested = False

    def start(self) -> dict:
        _safe_log("[CONTROLLER] start requested")
        status = self.get_status()

        if not self.lifecycle.boot_ready:
            _safe_log(
                f'[SESSION] start rejected {{ reason: "not_ready", '
                f'lifecycle_id: {self.lifecycle.lifecycle_id} }}'
            )
            return {
                "ok": False,
                "reason": "not_ready",
                "sessionActive": status["sessionActive"],
                "sessionState": status["sessionState"],
                "bootReady": status["bootReady"],
                "lifecycleId": status["lifecycleId"],
                "elapsed": status["elapsed"],
            }

        if self.lifecycle.session_active:
            _safe_log(
                f'[SESSION] start skipped {{ reason: "already_active", '
                f'lifecycle_id: {self.lifecycle.lifecycle_id} }}'
            )
            return {
                "ok": True,
                "reason": "already_active",
                "sessionActive": True,
                "sessionState": "active",
                "bootReady": True,
                "lifecycleId": status["lifecycleId"],
                "elapsed": status["elapsed"],
            }

        try:
            started = start_session(self.lifecycle)
        except Exception as error:
            _safe_log(
                f'[SESSION] start failed {{ reason: "start_failed", error: {error}, '
                f'lifecycle_id: {self.lifecycle.lifecycle_id} }}'
            )
            status = self.get_status()
            return {
                "ok": False,
                "reason": "start_failed",
                "sessionActive": status["sessionActive"],
                "sessionState": status["sessionState"],
                "bootReady": status["bootReady"],
                "lifecycleId": status["lifecycleId"],
                "elapsed": status["elapsed"],
                "error": str(error),
            }

        status = self.get_status()
        if not started:
            return {
                "ok": False,
                "reason": "start_failed",
                "sessionActive": status["sessionActive"],
                "sessionState": status["sessionState"],
                "bootReady": status["bootReady"],
                "lifecycleId": status["lifecycleId"],
                "elapsed": status["elapsed"],
            }

        return {
            "ok": True,
            "reason": "started",
            "sessionActive": True,
            "sessionState": "active",
            "bootReady": True,
            "lifecycleId": status["lifecycleId"],
            "elapsed": status["elapsed"],
        }

    def stop(self) -> dict:
        _safe_log("[CONTROLLER] stop requested")
        return end_session(self.lifecycle)

    def quit(self) -> None:
        _safe_log("[CONTROLLER] quit requested")
        self._shutdown_requested = True

    def is_active(self) -> bool:
        return self.lifecycle.session_active

    def mark_boot_ready(self) -> None:
        if not self.lifecycle.boot_ready:
            self.lifecycle.boot_ready = True
            _safe_log("[ANALYZER_READY] bootReady false -> true")

    def get_status(self) -> dict:
        if self.lifecycle.stop_in_progress:
            session_state = "stopping"
        elif self.lifecycle.session_active:
            session_state = "active"
        else:
            session_state = "idle"
        return {
            "running": True,
            "bootReady": self.lifecycle.boot_ready,
            "sessionActive": self.lifecycle.session_active,
            "elapsed": round(self.lifecycle.elapsed, 2),
            "shutdownRequested": self._shutdown_requested,
            "sessionState": session_state,
            "lifecycleId": self.lifecycle.lifecycle_id,
        }

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested


# ══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════

def main():
    lifecycle = SessionLifecycle()
    controller = AnalyzerController(lifecycle)
    start_analyzer_api_server(controller)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)

    movement_history = deque(maxlen=MOVEMENT_HISTORY)
    last_valid       = [0.5] * VECTOR_LEN

    # State smoothing buffers
    posture_buf    = StateBuffer(default="Calibrating")
    movement_buf   = StateBuffer(default="Calibrating")
    gaze_buf       = StateBuffer(default="Calibrating")
    expression_buf = StateBuffer(default="Calibrating")
    lateral_buf    = StateBuffer(default="Center")
    forward_buf    = StateBuffer(default="Neutral")
    gesture_buf    = StateBuffer(default="No Hands")

    # Score smoothing buffers
    sb_posture    = ScoreBuffer()
    sb_movement   = ScoreBuffer()
    sb_eye        = ScoreBuffer()
    sb_expression = ScoreBuffer()
    sb_gesture    = ScoreBuffer()
    sb_engagement = ScoreBuffer()
    sb_attention  = ScoreBuffer()

    prev_time = time.time()
    print("[INFO] Phase 4 Analyzer ready.")
    print("[INFO] [S] = Start session   [E] = End & generate report   [Q] = Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        controller.mark_boot_ready()

        frame    = cv2.flip(frame, 1)
        h, w     = frame.shape[:2]
        now_wall = time.time()

        if lifecycle.session_active:
            lifecycle.elapsed = now_wall - lifecycle.session_start_t
            if not lifecycle.loop_active_logged:
                lifecycle.loop_active_logged = True
                _safe_log(
                    f"[MAIN_LOOP] session active detected {{ elapsed: {lifecycle.elapsed:.2f}, "
                    f"lifecycle_id: {lifecycle.lifecycle_id} }}"
                )
            remaining = (max(INTERVIEW_DURATION - lifecycle.elapsed, 0.0)
                         if INTERVIEW_DURATION > 0 else 0.0)
            if INTERVIEW_DURATION > 0 and lifecycle.elapsed >= INTERVIEW_DURATION:
                _safe_log("[INFO] Time limit reached — generating report ...")
                end_session(lifecycle)
        else:
            remaining = float(INTERVIEW_DURATION)

        # ── MediaPipe inference ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        pose_res = POSE.process(rgb)
        face_res = FACE.process(rgb)
        hand_res = HANDS.process(rgb)
        rgb.flags.writeable = True

        # Defaults
        posture_status = "No Pose"
        composite      = 0.0
        lateral        = "Unknown"
        forward        = "Unknown"
        gaze           = "Unknown"
        expression     = "Unknown"
        movement       = "No Pose"
        gesture        = "No Hands"
        variance       = 0.0
        gaze_attention = 0.5
        eye_ar         = 0.25
        face_visible   = False
        brow_tension   = 0.0

        if pose_res.pose_landmarks:
            lms = pose_res.pose_landmarks.landmark
            posture_status, composite, _, _ = detect_posture(lms, w, h, lifecycle.calibrator)
            movement, variance = detect_movement(lms, w, h, movement_history, last_valid)
            mp_drawing.draw_landmarks(
                frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(80, 80, 220), thickness=2),
            )

        if face_res.multi_face_landmarks:
            face_lms            = face_res.multi_face_landmarks[0]
            face_visible        = True
            lateral, forward, _ = detect_head_tilt_face(face_lms, w, h)
            gaze, _, _, gaze_attention, eye_ar = detect_gaze(
                face_lms, w, h, lifecycle.gaze_ratio_buf)
            expression, _, _, brow_tension = detect_expression(face_lms, w, h)

        if hand_res.multi_hand_landmarks:
            for hand_lms in hand_res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 180), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 180, 100), thickness=2),
                )

        gesture, _, _ = detect_gestures(hand_res, w, h)

        # ── State smoothing ──
        posture_status = posture_buf.update(posture_status)
        movement       = movement_buf.update(movement)
        gaze           = gaze_buf.update(gaze)
        expression     = expression_buf.update(expression)
        lateral        = lateral_buf.update(lateral)
        forward        = forward_buf.update(forward)
        gesture        = gesture_buf.update(gesture)

        # ── Score calculation ──
        rp, rm, re, rx, rg, reng, ratt = calculate_scores(
            posture_status, movement, gaze, expression,
            gesture, composite, variance, gaze_attention, eye_ar)

        s_posture    = sb_posture.update(rp)
        s_movement   = sb_movement.update(rm)
        s_eye        = sb_eye.update(re)
        s_expression = sb_expression.update(rx)
        s_gesture    = sb_gesture.update(rg)
        s_engagement = sb_engagement.update(reng)
        s_attention  = sb_attention.update(ratt)
        overall_live = round(
            (s_posture + s_movement + s_eye + s_expression + s_gesture) / 5, 1)

        # ── Session logging & event detection ──
        if lifecycle.session_active:
            frame_log = FrameLog(
                time=lifecycle.elapsed,
                posture=posture_status,
                movement=movement,
                gaze=gaze,
                expression=expression,
                gesture=gesture,
                posture_score=s_posture,
                movement_score=s_movement,
                eye_score=s_eye,
                expression_score=s_expression,
                gesture_score=s_gesture,
                composite_angle=composite,
                variance=variance,
                gaze_attention=gaze_attention,
                eye_ar=eye_ar,
                engagement_score=s_engagement,
                attention_score=s_attention,
            )
            lifecycle.session_logger.try_log(lifecycle.elapsed, frame_log)
            lifecycle.timeline_tracker.record(
                lifecycle.elapsed, s_posture, s_movement, s_eye,
                s_expression, s_gesture, s_engagement, s_attention)
            lifecycle.event_detector.update(
                lifecycle.elapsed, gaze, posture_status, variance,
                eye_ar, face_visible, movement)

        warning     = lifecycle.event_detector.latest_warning(now=lifecycle.elapsed)
        event_count = len(lifecycle.event_detector.get_events())

        fps       = 1.0 / max(now_wall - prev_time, 1e-6)
        prev_time = now_wall

        data = {
            "posture_status": posture_status,
            "composite":      composite,
            "lateral":        lateral,
            "forward":        forward,
            "gaze":           gaze,
            "expression":     expression,
            "movement":       movement,
            "gesture":        gesture,
            "s_posture":      s_posture,
            "s_movement":     s_movement,
            "s_eye":          s_eye,
            "s_expression":   s_expression,
            "s_gesture":      s_gesture,
            "s_engagement":   s_engagement,
            "s_attention":    s_attention,
        }

        frame = draw_overlay(
            frame, data, fps,
            lifecycle.session_active, lifecycle.elapsed, remaining,
            overall_live, warning, event_count,
            lifecycle.calibrator,
            distraction_count=lifecycle.event_detector.distraction_count)

        cv2.imshow("AI Interview Behavior Analyzer — Phase 4", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('s'), ord('S')):
            controller.start()

        elif key in (ord('e'), ord('E')):
            controller.stop()

        elif key in (ord('q'), ord('Q')):
            print("[INFO] Quitting.")
            break

        if controller.shutdown_requested:
            print("[INFO] Shutdown requested via API.")
            break

    cap.release()
    cv2.destroyAllWindows()
    POSE.close()
    FACE.close()
    HANDS.close()


if __name__ == "__main__":
    main()
