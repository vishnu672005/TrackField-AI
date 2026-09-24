import cv2
import math
import numpy as np
import mediapipe as mp

# MediaPipe setup
LM = mp.solutions.pose.PoseLandmark
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    )

    return math.degrees(
        math.acos(np.clip(cosine, -1.0, 1.0))
    )


def extract_biomechanics(frame_bgr):

    h, w = frame_bgr.shape[:2]

    rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB
    )

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.4
    ) as pose:

        results = pose.process(rgb)

    if results.pose_landmarks is None:
        return None, rgb

    lms = results.pose_landmarks.landmark

    def pt(idx):
        return [
            lms[idx].x * w,
            lms[idx].y * h
        ]

    measurements = {}

    # Knee angle
    try:
        left_knee = calc_angle(
            pt(LM.LEFT_HIP),
            pt(LM.LEFT_KNEE),
            pt(LM.LEFT_ANKLE)
        )

        right_knee = calc_angle(
            pt(LM.RIGHT_HIP),
            pt(LM.RIGHT_KNEE),
            pt(LM.RIGHT_ANKLE)
        )

        measurements["knee_angle"] = min(
            left_knee,
            right_knee
        )

    except:
        measurements["knee_angle"] = 180.0

    # Hip angle
    try:
        left_hip = calc_angle(
            pt(LM.LEFT_SHOULDER),
            pt(LM.LEFT_HIP),
            pt(LM.LEFT_KNEE)
        )

        right_hip = calc_angle(
            pt(LM.RIGHT_SHOULDER),
            pt(LM.RIGHT_HIP),
            pt(LM.RIGHT_KNEE)
        )

        measurements["hip_angle"] = min(
            left_hip,
            right_hip
        )

    except:
        measurements["hip_angle"] = 180.0

    # Elbow angle
    try:
        left_elbow = calc_angle(
            pt(LM.LEFT_SHOULDER),
            pt(LM.LEFT_ELBOW),
            pt(LM.LEFT_WRIST)
        )

        right_elbow = calc_angle(
            pt(LM.RIGHT_SHOULDER),
            pt(LM.RIGHT_ELBOW),
            pt(LM.RIGHT_WRIST)
        )

        measurements["elbow_angle"] = min(
            left_elbow,
            right_elbow
        )

    except:
        measurements["elbow_angle"] = 180.0

    # Shoulder angle
    try:
        left_shoulder = calc_angle(
            pt(LM.LEFT_ELBOW),
            pt(LM.LEFT_SHOULDER),
            pt(LM.LEFT_HIP)
        )

        right_shoulder = calc_angle(
            pt(LM.RIGHT_ELBOW),
            pt(LM.RIGHT_SHOULDER),
            pt(LM.RIGHT_HIP)
        )

        measurements["shoulder_angle"] = max(
            left_shoulder,
            right_shoulder
        )

    except:
        measurements["shoulder_angle"] = 0.0

    # Trunk lean and spine lateral angle
    try:

        left_shoulder = [
            lms[LM.LEFT_SHOULDER].x * w,
            lms[LM.LEFT_SHOULDER].y * h
        ]

        right_shoulder = [
            lms[LM.RIGHT_SHOULDER].x * w,
            lms[LM.RIGHT_SHOULDER].y * h
        ]

        left_hip = [
            lms[LM.LEFT_HIP].x * w,
            lms[LM.LEFT_HIP].y * h
        ]

        right_hip = [
            lms[LM.RIGHT_HIP].x * w,
            lms[LM.RIGHT_HIP].y * h
        ]

        mid_shoulder = [
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2
        ]

        mid_hip = [
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2
        ]

        dx = mid_shoulder[0] - mid_hip[0]
        dy = mid_hip[1] - mid_shoulder[1]

        measurements["trunk_lean"] = abs(
            math.degrees(
                math.atan2(dx, dy + 1e-8)
            )
        )

        measurements["spine_lateral"] = abs(
            math.degrees(
                math.atan2(
                    abs(dx),
                    dy + 1e-8
                )
            )
        )

    except:

        measurements["trunk_lean"] = 0.0
        measurements["spine_lateral"] = 0.0

    # Draw pose landmarks
    annotated = rgb.copy()

    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=
            mp_drawing_styles.get_default_pose_landmarks_style()
    )

    return measurements, annotated


# --------------------------------------------------
# Test video
# --------------------------------------------------

VIDEO_PATH = "videos/javelinthrow.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("Total video frames:", total_frames)

sample_every = max(
    1,
    total_frames // 24
)

print(
    "Running MediaPipe approximately every",
    sample_every,
    "frames..."
)

all_measurements = []

frame_index = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_index % sample_every == 0:

        measurements, annotated = \
            extract_biomechanics(frame)

        if measurements is not None:

            # Same validation used in notebook
            if (
                30 < measurements.get(
                    "knee_angle", 0
                ) < 195
                and
                20 < measurements.get(
                    "hip_angle", 0
                ) < 195
                and
                measurements.get(
                    "trunk_lean", 0
                ) < 85
            ):

                all_measurements.append(
                    measurements
                )

    frame_index += 1

cap.release()


print("\n" + "=" * 50)
print("MEDIAPIPE RESULT")
print("=" * 50)

print(
    "Valid pose frames:",
    len(all_measurements)
)

if all_measurements:

    for key in [
        "knee_angle",
        "hip_angle",
        "elbow_angle",
        "shoulder_angle",
        "trunk_lean",
        "spine_lateral"
    ]:

        values = [
            m[key]
            for m in all_measurements
            if key in m
        ]

        print(
            f"{key:18s}: "
            f"average = {np.mean(values):.1f}°   "
            f"minimum = {np.min(values):.1f}°   "
            f"maximum = {np.max(values):.1f}°"
        )

else:

    print("No valid poses detected.")

print("=" * 50)