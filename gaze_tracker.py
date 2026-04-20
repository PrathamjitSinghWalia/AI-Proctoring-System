import cv2
import numpy as np
import math
import mediapipe as mp

# ============================
# CONSTANTS - LANDMARK INDEXES
# ============================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

LEFT_EYE_LEFT_CORNER = 362
LEFT_EYE_RIGHT_CORNER = 263
RIGHT_EYE_LEFT_CORNER = 133
RIGHT_EYE_RIGHT_CORNER = 33

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

NOSE_TIP = 4
CHIN = 152
LEFT_TEMPLE = 234
RIGHT_TEMPLE = 454
FOREHEAD = 10

# ============================
# INIT MEDIAPIPE
# ============================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ============================
# FUNCTIONS
# ============================

def get_eye_ratio(landmarks, eye_points, fw, fh):
    """Calculate Eye Aspect Ratio (EAR)"""
    points = []
    for idx in eye_points:
        x = int(landmarks[idx].x * fw)
        y = int(landmarks[idx].y * fh)
        points.append((x, y))

    vertical_1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    vertical_2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    horizontal = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def get_iris_position(landmarks, eye_left_corner, eye_right_corner, iris_points, fw, fh):
    """
    Calculate where the iris sits inside the eye horizontally.
    0 = iris at left corner (looking left)
    0.5 = iris in center (looking forward)
    1 = iris at right corner (looking right)
    """
    left_corner = (landmarks[eye_left_corner].x * fw,
                   landmarks[eye_left_corner].y * fh)
    right_corner = (landmarks[eye_right_corner].x * fw,
                    landmarks[eye_right_corner].y * fh)

    iris_x = sum(landmarks[p].x for p in iris_points) / len(iris_points)
    iris_y = sum(landmarks[p].y for p in iris_points) / len(iris_points)
    iris_center = (iris_x * fw, iris_y * fh)

    eye_width = math.sqrt((right_corner[0] - left_corner[0])**2 +
                          (right_corner[1] - left_corner[1])**2)

    iris_to_left = math.sqrt((iris_center[0] - left_corner[0])**2 +
                             (iris_center[1] - left_corner[1])**2)

    if eye_width == 0:
        return 0.5

    ratio = iris_to_left / eye_width
    return ratio


def get_head_pose(landmarks, fw, fh):
    """
    Get head direction ratios.
    Uses forehead as fixed reference point for vertical calculation.
    This avoids the math identity bug where v_ratio was always -0.5
    """
    nose = landmarks[NOSE_TIP]
    chin = landmarks[CHIN]
    left_temple = landmarks[LEFT_TEMPLE]
    right_temple = landmarks[RIGHT_TEMPLE]
    forehead = landmarks[FOREHEAD]

    # Convert all points to pixels
    nose_x = nose.x * fw
    nose_y = nose.y * fh
    chin_y = chin.y * fh
    forehead_y = forehead.y * fh
    left_x = left_temple.x * fw
    right_x = right_temple.x * fw

    # Horizontal ratio
    # 0 = centered, negative = looking left, positive = looking right
    face_center_x = (left_x + right_x) / 2
    face_width = right_x - left_x
    h_ratio = (nose_x - face_center_x) / face_width if face_width != 0 else 0

    # Vertical ratio
    # Measures where nose sits between forehead and chin
    # looking down = nose closer to chin = higher ratio
    # looking up = nose closer to forehead = lower ratio
    # neutral = nose roughly in middle = around 0.45-0.55
    total_face_height = chin_y - forehead_y
    v_ratio = (nose_y - forehead_y) / total_face_height if total_face_height != 0 else 0.5

    return h_ratio, v_ratio


def get_gaze_direction(landmarks, fw, fh):
    """Combined head pose + iris tracking for robust gaze detection"""
    h_ratio, v_ratio = get_head_pose(landmarks, fw, fh)

    left_iris_ratio = get_iris_position(
        landmarks,
        LEFT_EYE_LEFT_CORNER,
        LEFT_EYE_RIGHT_CORNER,
        LEFT_IRIS, fw, fh
    )
    right_iris_ratio = get_iris_position(
        landmarks,
        RIGHT_EYE_LEFT_CORNER,
        RIGHT_EYE_RIGHT_CORNER,
        RIGHT_IRIS, fw, fh
    )

    avg_iris_ratio = (left_iris_ratio + right_iris_ratio) / 2

    # Head horizontal - tuned from your real data
    # your forward H was ~-0.008, looking left was ~-0.44
    head_horizontal = None
    if h_ratio < -0.20:
        head_horizontal = "LEFT"
    elif h_ratio > 0.20:
        head_horizontal = "RIGHT"

    iris_horizontal = None
    if avg_iris_ratio < 0.530:
        iris_horizontal = "LEFT"
    elif avg_iris_ratio > 0.580:
        iris_horizontal = "RIGHT"

    # Vertical - thresholds will be tuned after seeing new V values
    # v_ratio is now between 0 and 1 (not stuck at -0.5)
    # neutral is around 0.45-0.55
    if v_ratio < 0.38:
        vertical = "UP"
    elif v_ratio > 0.58:
        vertical = "DOWN"
    else:
        vertical = None

    # Combine both signals
    if vertical:
        return f"Looking {vertical}"
    elif head_horizontal or iris_horizontal:
        direction = head_horizontal or iris_horizontal
        return f"Looking {direction}"
    else:
        return "Looking FORWARD"


