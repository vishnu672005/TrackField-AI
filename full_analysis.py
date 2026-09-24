
# Track & Field local end-to-end analysis
# Adapted directly from the uploaded Kaggle notebook.
# Local checkpoint: model/videomae_trackfield_final.pth

import os
import cv2
import math
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import mediapipe as mp

from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)

warnings.filterwarnings("ignore")

BASE = os.path.abspath("outputs")
os.makedirs(BASE, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
MODEL_PATH = "model/videomae_trackfield_final.pth"

LABEL_TO_EVENT = {
    0: "Long Jump",
    1: "High Jump",
    2: "Pole Vault",
    3: "Shot Put",
    4: "Discus Throw",
    5: "Javelin Throw",
    6: "Sprint",
}
NUM_CLASSES = 7
EVENT_NAMES = [LABEL_TO_EVENT[i] for i in range(NUM_CLASSES)]

LM = mp.solutions.pose.PoseLandmark
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

print(f"Device: {DEVICE}")
print("Loading VideoMAE processor...")
processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)

print("Loading VideoMAE model...")
model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)
state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model = model.to(DEVICE)
model.eval()

print("VideoMAE trained weights loaded successfully.")


def extract_frames(video_path, num_frames=16):
    """
    Extract exactly 16 evenly spaced frames from a video.
    VideoMAE requires exactly 16 frames as input.
    Returns list of 16 RGB numpy arrays, or None if failed.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < num_frames:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames

#13
# ── Re-define MediaPipe globals ───────────────────────────────────
import mediapipe as mp
import math
LM                = mp.solutions.pose.PoseLandmark
mp_pose           = mp.solutions.pose
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ── Event type mapping ────────────────────────────────────────────
EVENT_TYPE_MAP = {
    'Long Jump':    'jumping',
    'High Jump':    'high_jump',
    'Pole Vault':   'pole_vault',
    'Shot Put':     'throwing',
    'Discus Throw': 'throwing',
    'Javelin Throw':'throwing',
    'Sprint':       'sprinting',
}

# ── Safe ranges ───────────────────────────────────────────────────
SAFE_RANGES = {
    'knee_angle': {
        'jumping':    (130, 180),
        'high_jump':  (120, 175),
        'pole_vault': (100, 175),
        'throwing':   (120, 170),
        'sprinting':  (130, 180),
    },
    'hip_angle': {
        'jumping':    (100, 180),
        'high_jump':  (130, 180),
        'pole_vault': (80,  175),
        'throwing':   (90,  160),
        'sprinting':  (150, 180),
    },
    'elbow_angle': {
        'jumping':    (60,  180),
        'high_jump':  (60,  180),
        'pole_vault': (60,  180),
        'throwing':   (70,  145),
        'sprinting':  (70,  110),
    },
    'shoulder_angle': {
        'jumping':    (0,   70),
        'high_jump':  (0,   80),
        'pole_vault': (60,  140),
        'throwing':   (70,  130),
        'sprinting':  (0,   65),
    },
    'trunk_lean': {
        'jumping':    (0,   30),
        'high_jump':  (0,   45),
        'pole_vault': (0,   50),
        'throwing':   (0,   45),
        'sprinting':  (0,   15),
    },
    'spine_lateral': {
        'jumping':    (0,   12),
        'high_jump':  (0,   40),
        'pole_vault': (0,   35),
        'throwing':   (0,   30),
        'sprinting':  (0,    8),
    },
}

# ── Injury rules ──────────────────────────────────────────────────
INJURY_RULES = {
    'jumping': [
        ("Knee valgus on landing",
         'knee_angle', lambda a: a < 150,
         "HIGH", "ACL / Meniscus tear risk", "Knee"),
        ("Excessive forward trunk lean",
         'trunk_lean', lambda a: a > 25,
         "MEDIUM", "Lower back strain risk", "Spine"),
        ("Hip drop at takeoff",
         'hip_angle', lambda a: a < 130,
         "HIGH", "Hip flexor / IT Band injury risk", "Hip"),
        ("Lateral spine deviation",
         'spine_lateral', lambda a: a > 12,
         "HIGH", "Lumbar stress fracture risk", "Spine"),
    ],
    'high_jump': [
        ("Insufficient back arch",
         'spine_lateral', lambda a: a < 15,
         "MEDIUM", "Reduced clearance, landing risk", "Spine"),
        ("Knee not driven high enough",
         'knee_angle', lambda a: a > 160,
         "MEDIUM", "Hamstring strain risk", "Knee"),
        ("Excessive trunk lean forward",
         'trunk_lean', lambda a: a > 30,
         "HIGH", "Lower back injury risk", "Spine"),
    ],
    'pole_vault': [
        ("Elbow below shoulder at plant",
         'elbow_angle', lambda a: a < 90,
         "HIGH", "Shoulder / UCL injury risk", "Elbow"),
        ("Insufficient hip drive",
         'hip_angle', lambda a: a > 150,
         "MEDIUM", "Reduced vault height", "Hip"),
        ("Excessive lateral spine bend",
         'spine_lateral', lambda a: a > 25,
         "HIGH", "Lumbar disc injury risk", "Spine"),
    ],
    'throwing': [
        ("Dropped elbow at release",
         'elbow_angle', lambda a: a < 85,
         "HIGH", "UCL / Tommy John injury risk", "Elbow"),
        ("Excessive trunk lateral bend",
         'spine_lateral', lambda a: a > 30,
         "HIGH", "Oblique / lumbar disc injury risk", "Spine"),
        ("Over-rotation of shoulder",
         'shoulder_angle', lambda a: a > 120,
         "HIGH", "Rotator cuff tear risk", "Shoulder"),
        ("Knee hyperextension",
         'knee_angle', lambda a: a > 170,
         "MEDIUM", "Knee hyperextension risk", "Knee"),
    ],
    'sprinting': [
        ("Excessive trunk lean",
         'trunk_lean', lambda a: a > 15,
         "MEDIUM", "Lower back overuse injury risk", "Spine"),
        ("Reduced knee lift",
         'knee_angle', lambda a: a < 130,
         "HIGH", "Hamstring strain risk", "Knee"),
        ("Hip drop during stance",
         'hip_angle', lambda a: a < 155,
         "HIGH", "IT Band / hip injury risk", "Hip"),
        ("Arm crossover",
         'elbow_angle', lambda a: a < 70,
         "LOW", "Shoulder tension, energy waste", "Shoulder"),
    ],
}

RISK_COLORS = {
    'HIGH':   '#e74c3c',
    'MEDIUM': '#f39c12',
    'LOW':    '#27ae60',
}

print("✅ Rules loaded")


# ── Helpers ───────────────────────────────────────────────────────
def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))


def frame_quality_score(frame_rgb):
    """
    Score how good a frame is for pose analysis.
    Higher = athlete is larger and clearer in frame.
    Uses edge density in centre region.
    """
    h, w   = frame_rgb.shape[:2]
    cy1, cy2 = int(h*0.15), int(h*0.85)
    cx1, cx2 = int(w*0.15), int(w*0.85)
    centre   = frame_rgb[cy1:cy2, cx1:cx2]
    gray     = cv2.cvtColor(centre, cv2.COLOR_RGB2GRAY)
    edges    = cv2.Canny(gray, 50, 150)
    return float(edges.sum())


def extract_biomechanics(frame_bgr):
    """Run MediaPipe on one frame. Returns (measurements, annotated_rgb)."""
    h, w = frame_bgr.shape[:2]
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.4
    ) as pose:
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

    if results.pose_landmarks is None:
        return None, rgb

    lms = results.pose_landmarks.landmark

    def pt(idx):
        return [lms[idx].x * w, lms[idx].y * h]

    m = {}
    try:
        lk = calc_angle(pt(LM.LEFT_HIP),
                        pt(LM.LEFT_KNEE),   pt(LM.LEFT_ANKLE))
        rk = calc_angle(pt(LM.RIGHT_HIP),
                        pt(LM.RIGHT_KNEE),  pt(LM.RIGHT_ANKLE))
        m['knee_angle'] = min(lk, rk)
    except: m['knee_angle'] = 180.0

    try:
        lh = calc_angle(pt(LM.LEFT_SHOULDER),
                        pt(LM.LEFT_HIP),    pt(LM.LEFT_KNEE))
        rh = calc_angle(pt(LM.RIGHT_SHOULDER),
                        pt(LM.RIGHT_HIP),   pt(LM.RIGHT_KNEE))
        m['hip_angle'] = min(lh, rh)
    except: m['hip_angle'] = 180.0

    try:
        le = calc_angle(pt(LM.LEFT_SHOULDER),
                        pt(LM.LEFT_ELBOW),  pt(LM.LEFT_WRIST))
        re = calc_angle(pt(LM.RIGHT_SHOULDER),
                        pt(LM.RIGHT_ELBOW), pt(LM.RIGHT_WRIST))
        m['elbow_angle'] = min(le, re)
    except: m['elbow_angle'] = 180.0

    try:
        ls = calc_angle(pt(LM.LEFT_ELBOW),
                        pt(LM.LEFT_SHOULDER),  pt(LM.LEFT_HIP))
        rs = calc_angle(pt(LM.RIGHT_ELBOW),
                        pt(LM.RIGHT_SHOULDER), pt(LM.RIGHT_HIP))
        m['shoulder_angle'] = max(ls, rs)
    except: m['shoulder_angle'] = 0.0

    try:
        l_sh = [lms[LM.LEFT_SHOULDER].x*w,
                lms[LM.LEFT_SHOULDER].y*h]
        r_sh = [lms[LM.RIGHT_SHOULDER].x*w,
                lms[LM.RIGHT_SHOULDER].y*h]
        l_hp = [lms[LM.LEFT_HIP].x*w,
                lms[LM.LEFT_HIP].y*h]
        r_hp = [lms[LM.RIGHT_HIP].x*w,
                lms[LM.RIGHT_HIP].y*h]
        mid_sh = [(l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2]
        mid_hp = [(l_hp[0]+r_hp[0])/2, (l_hp[1]+r_hp[1])/2]
        dx = mid_sh[0] - mid_hp[0]
        dy = mid_hp[1] - mid_sh[1]
        m['trunk_lean']    = abs(
            math.degrees(math.atan2(dx, dy + 1e-8)))
        m['spine_lateral'] = abs(
            math.degrees(math.atan2(abs(dx), dy + 1e-8)))
    except:
        m['trunk_lean']    = 0.0
        m['spine_lateral'] = 0.0

    annotated = rgb.copy()
    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=
            mp_drawing_styles.get_default_pose_landmarks_style()
    )
    return m, annotated


def analyse_injury_risk(measurements, event_type):
    rules    = INJURY_RULES.get(event_type,
                                INJURY_RULES['jumping'])
    findings = []
    for desc, key, condition, severity, detail, body_part \
            in rules:
        val = measurements.get(key, None)
        if val is not None and condition(val):
            findings.append({
                'description': desc,
                'body_part':   body_part,
                'severity':    severity,
                'detail':      detail,
                'value':       round(val, 1),
            })
    findings.sort(
        key=lambda x: {'HIGH':0,'MEDIUM':1,'LOW':2}.get(
            x['severity'], 3))
    return findings


def overall_risk_score(findings):
    score = 0
    for f in findings:
        if f['severity'] == 'HIGH':     score += 30
        elif f['severity'] == 'MEDIUM': score += 15
        elif f['severity'] == 'LOW':    score += 5
    return min(score, 100)


def is_sprinting_biomechanics(meas_list):
    """
    Sprinting signature vs Long Jump:
    - Sprinting : upright trunk ALL the time, knee never deeply bent,
                  consistent rhythm across ALL frames
    - Long Jump  : has a phase where hip/knee bends sharply (takeoff/landing)
                  so min values drop significantly
    """
    if not meas_list:
        return False, 0.0

    knee_vals   = [m.get('knee_angle',    180) for m in meas_list]
    hip_vals    = [m.get('hip_angle',     180) for m in meas_list]
    trunk_vals  = [m.get('trunk_lean',      0) for m in meas_list]
    lateral_vals= [m.get('spine_lateral',   0) for m in meas_list]
    elbow_vals  = [m.get('elbow_angle',   180) for m in meas_list]

    avg_knee    = np.mean(knee_vals)
    min_knee    = np.min(knee_vals)
    avg_hip     = np.mean(hip_vals)
    min_hip     = np.min(hip_vals)
    avg_trunk   = np.mean(trunk_vals)
    avg_lateral = np.mean(lateral_vals)
    avg_elbow   = np.mean(elbow_vals)

    print(f"\n   Biomechanics detail:")
    print(f"     avg knee    : {avg_knee:.1f}°  min: {min_knee:.1f}°")
    print(f"     avg hip     : {avg_hip:.1f}°  min: {min_hip:.1f}°")
    print(f"     avg trunk   : {avg_trunk:.1f}°")
    print(f"     avg lateral : {avg_lateral:.1f}°")
    print(f"     avg elbow   : {avg_elbow:.1f}°")

    # ── Sprint indicators ─────────────────────────────────────────
    sprint_score = 0

    # Upright trunk throughout — most important sprint indicator
    if avg_trunk < 15:              sprint_score += 30
    elif avg_trunk < 20:            sprint_score += 15

    # Knee never deeply bent — in long jump, min_knee drops below 120
    # In sprinting, knee stays above 130 even at highest lift
    if min_knee > 130:              sprint_score += 30
    elif min_knee > 120:            sprint_score += 15

    # Hip never deeply flexed — long jump has deep hip flexion in flight
    if min_hip > 140:               sprint_score += 20
    elif min_hip > 130:             sprint_score += 10

    # Minimal lateral spine bend — sprinting is very upright laterally
    if avg_lateral < 8:             sprint_score += 10

    # Elbow pumping at ~90° — sprint arm drive
    if 70 < avg_elbow < 120:        sprint_score += 10

    # ── Long jump indicators (if these present = NOT sprinting) ───
    long_jump_score = 0

    # Deep knee bend at takeoff/landing
    if min_knee < 120:              long_jump_score += 40
    elif min_knee < 130:            long_jump_score += 20

    # Deep hip flexion in flight
    if min_hip < 110:               long_jump_score += 40
    elif min_hip < 130:             long_jump_score += 20

    # Forward trunk lean
    if avg_trunk > 20:              long_jump_score += 20

    print(f"     sprint_score    : {sprint_score}")
    print(f"     long_jump_score : {long_jump_score}")

    # Sprint wins only if sprint score clearly beats long jump score
    is_sprint = (sprint_score >= 60 and
                 sprint_score > long_jump_score + 10)
    ratio = sprint_score / 100.0

    return is_sprint, ratio


def is_discus_vs_shotput(meas_list):
    """
    Discus: more lateral rotation, higher shoulder elevation,
            wider stance
    Shot Put: more linear push, elbow higher, less rotation
    Returns 'Discus Throw' or 'Shot Put'
    """
    if not meas_list:
        return None
    avg_lateral  = np.mean([m.get('spine_lateral', 0)
                             for m in meas_list])
    avg_shoulder = np.mean([m.get('shoulder_angle', 0)
                             for m in meas_list])
    avg_elbow    = np.mean([m.get('elbow_angle', 180)
                             for m in meas_list])

    # Discus has MORE lateral rotation and shoulder turn
    # Shot put has higher elbow (pushing motion)
    discus_score  = 0
    shotput_score = 0

    if avg_lateral  > 25:  discus_score  += 40
    else:                  shotput_score += 30
    if avg_shoulder > 90:  discus_score  += 35
    else:                  shotput_score += 25
    if avg_elbow    < 100: shotput_score += 30
    else:                  discus_score  += 20

    return ('Discus Throw' if discus_score > shotput_score
            else 'Shot Put')


def generate_full_report(measurements, findings,
                          event, event_type,
                          videomae_confidence,
                          best_frame_rgb,
                          source_name=""):
    risk_score = overall_risk_score(findings)
    n_findings = len(findings)
    fig_height = 16 + max(0, n_findings - 2) * 1.8
    fig        = plt.figure(figsize=(18, fig_height))
    fig.patch.set_facecolor('#0f0f1a')

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        4, 2, figure=fig,
        left=0.05, right=0.97,
        top=0.91,  bottom=0.03,
        hspace=0.6, wspace=0.35,
        height_ratios=[1.2, 1.0, 1.8,
                       max(1.0, n_findings * 1.0)]
    )

    fig.text(0.5, 0.965,
             '🏃  Track & Field — Biomechanical Injury Risk Report',
             ha='center', fontsize=20,
             fontweight='bold', color='white')
    fig.text(0.5, 0.938,
             f'Event: {event}   |   '
             f'Model Confidence: {videomae_confidence:.1f}%   |   '
             f'Source: {source_name}',
             ha='center', fontsize=12, color='#aaaacc')

    # ── Best frame ────────────────────────────────────────────────
    ax_frame = fig.add_subplot(gs[0, :])
    ax_frame.imshow(best_frame_rgb)
    ax_frame.set_title(
        'Key Frame — Pose Detection '
        '(highest quality frame selected)',
        color='white', fontsize=11,
        fontweight='bold', pad=8)
    ax_frame.axis('off')

    # ── Risk gauge ────────────────────────────────────────────────
    ax_gauge = fig.add_subplot(gs[1, 0])
    ax_gauge.set_facecolor('#16213e')
    gauge_color = ('#27ae60' if risk_score < 30 else
                   '#f39c12' if risk_score < 60 else '#e74c3c')
    ax_gauge.barh([''], [100],
                  color='#2c2c4e', height=0.5)
    ax_gauge.barh([''], [risk_score],
                  color=gauge_color, height=0.5)
    ax_gauge.text(
        min(risk_score - 2, 88), 0,
        f'{risk_score}/100',
        va='center', ha='right',
        color='white', fontsize=20, fontweight='bold'
    )
    risk_label = ('LOW RISK'      if risk_score < 30 else
                  'MODERATE RISK' if risk_score < 60 else
                  'HIGH RISK')
    ax_gauge.text(50, -0.42, risk_label,
                  va='top', ha='center',
                  color=gauge_color,
                  fontsize=13, fontweight='bold')
    ax_gauge.set_xlim(0, 100)
    ax_gauge.set_ylim(-0.6, 0.6)
    ax_gauge.set_title('Overall Injury Risk Score',
                        color='white', fontsize=12,
                        fontweight='bold', pad=10)
    ax_gauge.tick_params(colors='#888')
    ax_gauge.set_xlabel('0 = Safe  →  100 = Critical',
                         color='#888', fontsize=9)
    for sp in ax_gauge.spines.values():
        sp.set_edgecolor('#334')

    # ── Summary ───────────────────────────────────────────────────
    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.set_facecolor('#16213e')
    ax_info.axis('off')
    ax_info.set_title('Analysis Summary',
                       color='white', fontsize=12,
                       fontweight='bold', pad=10)
    rows = [
        ('Event',       event),
        ('Category',    event_type.replace('_',' ').capitalize()),
        ('Confidence',  f'{videomae_confidence:.1f}%'),
        ('Risk Score',  f'{risk_score}/100'),
        ('Total Flags', str(n_findings)),
        ('High Risk',   str(sum(1 for f in findings
                                if f['severity']=='HIGH'))),
        ('Medium Risk', str(sum(1 for f in findings
                                if f['severity']=='MEDIUM'))),
        ('Low Risk',    str(sum(1 for f in findings
                                if f['severity']=='LOW'))),
    ]
    for i, (label, value) in enumerate(rows):
        y = 0.92 - i * 0.115
        ax_info.text(0.02, y, f'{label}:',
                     transform=ax_info.transAxes,
                     fontsize=11, color='#aaaacc', va='top')
        ax_info.text(0.50, y, value,
                     transform=ax_info.transAxes,
                     fontsize=11, color='white',
                     fontweight='bold', va='top')
    for sp in ax_info.spines.values():
        sp.set_edgecolor('#334')

    # ── Joint angles ──────────────────────────────────────────────
    ax_j = fig.add_subplot(gs[2, :])
    ax_j.set_facecolor('#16213e')
    jkeys  = ['knee_angle','hip_angle','elbow_angle',
               'shoulder_angle','trunk_lean','spine_lateral']
    jlabels= ['Knee\nAngle°','Hip\nAngle°','Elbow\nAngle°',
               'Shoulder\nAngle°','Trunk\nLean°','Spine\nLateral°']
    jvals  = [measurements.get(k, 0) for k in jkeys]
    smins  = [SAFE_RANGES[k].get(event_type,(0,180))[0]
               for k in jkeys]
    smaxs  = [SAFE_RANGES[k].get(event_type,(0,180))[1]
               for k in jkeys]
    bcolors= ['#e74c3c' if (v<mn or v>mx) else '#2ecc71'
               for v,mn,mx in zip(jvals,smins,smaxs)]
    xp = np.arange(len(jkeys))
    bars = ax_j.bar(xp, jvals, color=bcolors,
                    edgecolor='#222', linewidth=0.8,
                    width=0.55, zorder=3)
    for bar, val in zip(bars, jvals):
        ax_j.text(bar.get_x()+bar.get_width()/2,
                  bar.get_height()+2,
                  f'{val:.0f}°',
                  ha='center', va='bottom',
                  color='white', fontsize=11,
                  fontweight='bold')
    for i,(mn,mx) in enumerate(zip(smins,smaxs)):
        ax_j.plot([i-.3,i+.3],[mn,mn],'--',
                   color='#f1c40f',lw=1.5,alpha=0.8,zorder=4)
        ax_j.plot([i-.3,i+.3],[mx,mx],'--',
                   color='#f1c40f',lw=1.5,alpha=0.8,zorder=4)
    ax_j.set_xticks(xp)
    ax_j.set_xticklabels(jlabels,
                          fontsize=11, color='white')
    ax_j.set_ylabel('Angle (degrees)',
                     color='#aaaacc', fontsize=11)
    ax_j.set_title(
        'Joint Angle Measurements   '
        '(🟢 Safe   🔴 Risk   - - Safe Range Limits)',
        color='white', fontsize=12,
        fontweight='bold', pad=12)
    ax_j.tick_params(colors='#888')
    ax_j.set_ylim(0, max(jvals)*1.25+20)
    ax_j.grid(axis='y', color='#334',
               linewidth=0.5, alpha=0.5)
    for sp in ax_j.spines.values():
        sp.set_edgecolor('#334')

    # ── Findings table ────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[3, :])
    ax_t.set_facecolor('#16213e')
    ax_t.axis('off')
    ax_t.set_title('⚠️   Biomechanical Risk Findings',
                    color='white', fontsize=13,
                    fontweight='bold', loc='left', pad=14)

    if not findings:
        ax_t.text(0.5, 0.5,
                  '✅  No significant injury risk detected.',
                  ha='center', va='center',
                  fontsize=14, color='#2ecc71',
                  transform=ax_t.transAxes)
    else:
        cols  = [('#',0.01),('Severity',0.06),
                 ('Body Part',0.18),('Finding',0.30),
                 ('Measured',0.76),('Injury Risk',0.86)]
        row_h = min(0.82/max(n_findings,1), 0.20)

        for label,x in cols:
            ax_t.text(x, 0.93, label,
                      transform=ax_t.transAxes,
                      fontsize=11, color='#aaaacc',
                      fontweight='bold', va='top')
        ax_t.axhline(y=0.88, xmin=0.01, xmax=0.99,
                      color='#445', linewidth=1.2)

        for i, f in enumerate(findings):
            y     = 0.85 - i * row_h
            color = RISK_COLORS.get(f['severity'], 'white')
            ax_t.text(cols[0][1], y, str(i+1),
                      transform=ax_t.transAxes,
                      fontsize=11, color='#888', va='top')
            ax_t.text(cols[1][1], y, f'● {f["severity"]}',
                      transform=ax_t.transAxes,
                      fontsize=11, color=color,
                      fontweight='bold', va='top')
            ax_t.text(cols[2][1], y, f['body_part'],
                      transform=ax_t.transAxes,
                      fontsize=11, color='white', va='top')
            ax_t.text(cols[3][1], y, f['description'],
                      transform=ax_t.transAxes,
                      fontsize=10, color='#dddddd', va='top')
            ax_t.text(cols[4][1], y, f'{f["value"]:.1f}°',
                      transform=ax_t.transAxes,
                      fontsize=11, color=color,
                      fontweight='bold', va='top')
            ax_t.text(cols[5][1], y, f['detail'],
                      transform=ax_t.transAxes,
                      fontsize=10, color=color, va='top')
            ax_t.axhline(y=y-row_h*0.35,
                          xmin=0.01, xmax=0.99,
                          color='#334', linewidth=0.6)

    plt.savefig(f'{BASE}/injury_risk_report.png',
                dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    plt.show()
    print(f"✅ Report saved → {BASE}/injury_risk_report.png")


# ── MAIN FUNCTION ─────────────────────────────────────────────────
def predict_and_analyse(video_path):
    filename = os.path.basename(video_path)
    print(f"\n{'='*55}")
    print(f"  Analysing: {filename}")
    print(f"{'='*55}\n")

    # ── Step A: Extract frames for VideoMAE ──────────────────────
    print("Step 1/4 — Extracting frames ...")
    frames_rgb = extract_frames(video_path, num_frames=16)
    if frames_rgb is None:
        print("❌ Could not extract frames")
        return

    # ── Step B: VideoMAE event classification ────────────────────
    print("Step 2/4 — VideoMAE: Classifying event ...")
    inputs = processor(images=frames_rgb, return_tensors="pt")
    pv     = inputs['pixel_values'].to(DEVICE)
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pv)

    probs          = torch.softmax(outputs.logits, dim=1
                                   ).squeeze().cpu().numpy()
    pred_idx       = int(probs.argmax())
    pred_event     = LABEL_TO_EVENT[pred_idx]
    mae_confidence = float(probs[pred_idx]) * 100
    event_type     = EVENT_TYPE_MAP.get(pred_event, 'jumping')

    print(f"   VideoMAE says : {pred_event} "
          f"({mae_confidence:.1f}%)")

    # ── Step C: MediaPipe biomechanics on ALL sampled frames ──────
    print("Step 3/4 — MediaPipe: Extracting biomechanics ...")
    cap          = cv2.VideoCapture(video_path)
    total_f      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_every = max(1, total_f // 24)

    all_meas      = []
    all_annotated = []
    all_raw_rgb   = []
    frame_idx     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            meas, ann = extract_biomechanics(frame)
            raw_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if meas is not None:
                # Validate angles are physically plausible
                if (30 < meas.get('knee_angle', 0) < 195 and
                    20 < meas.get('hip_angle',  0) < 195 and
                    meas.get('trunk_lean', 0) < 85):
                    all_meas.append(meas)
                    all_annotated.append(ann)
                    all_raw_rgb.append(raw_rgb)
        frame_idx += 1
    cap.release()

    print(f"   Valid frames  : {len(all_meas)}")

    if not all_meas:
        print("❌ No valid poses detected")
        return

    # ── Fix 1: Sprint biomechanics override ──────────────────────
    is_sprint, sprint_ratio = is_sprinting_biomechanics(all_meas)
    print(f"   Sprint detected  : {is_sprint} "
          f"(score={sprint_ratio*100:.0f})")

    if is_sprint:
        print(f"   → Biomechanics override → Sprint")
        pred_event     = 'Sprint'
        event_type     = 'sprinting'
        mae_confidence = sprint_ratio * 100

    elif pred_event == 'Long Jump':
        # Double check — long jump must have a deep flex phase
        knee_vals = [m.get('knee_angle', 180) for m in all_meas]
        hip_vals  = [m.get('hip_angle',  180) for m in all_meas]
        min_knee  = min(knee_vals)
        min_hip   = min(hip_vals)

        if min_knee > 135 and min_hip > 145:
            print(f"   → Long Jump overridden → Sprint")
            print(f"     (no deep flex phase found: "
                  f"min_knee={min_knee:.1f}° "
                  f"min_hip={min_hip:.1f}°)")
            pred_event     = 'Sprint'
            event_type     = 'sprinting'
            mae_confidence = 70.0
        else:
            print(f"   → Long Jump confirmed "
                  f"(min_knee={min_knee:.1f}° "
                  f"min_hip={min_hip:.1f}°)")

    # ── Fix 2: Discus vs Shot Put disambiguation ─────────────────
    if pred_event in ('Discus Throw', 'Shot Put'):
        refined = is_discus_vs_shotput(all_meas)
        if refined and refined != pred_event:
            print(f"   → Throwing refinement: "
                  f"{pred_event} → {refined}")
            pred_event = refined
            event_type = 'throwing'

    print(f"   Final event   : {pred_event}")

    # ── Fix 3: Best frame = highest quality frame ─────────────────
    if all_annotated:
        scores    = [frame_quality_score(f)
                     for f in all_raw_rgb]
        best_idx  = int(np.argmax(scores))
        best_frame = all_annotated[best_idx]
        print(f"   Best frame    : #{best_idx} "
              f"(score={scores[best_idx]:.0f})")
    else:
        best_frame = all_annotated[0]

    # ── Step D: Injury risk analysis ─────────────────────────────
    print("Step 4/4 — Analysing injury risk ...")
    avg_meas = {
        k: float(np.mean([m[k] for m in all_meas if k in m]))
        for k in all_meas[0]
    }
    findings = analyse_injury_risk(avg_meas, event_type)
    risk     = overall_risk_score(findings)

    print(f"\n{'='*55}")
    print(f"  FINAL EVENT   : {pred_event}")
    print(f"  CONFIDENCE    : {mae_confidence:.1f}%")
    print(f"  RISK SCORE    : {risk}/100")
    print(f"  FLAGS RAISED  : {len(findings)}")
    for f in findings:
        print(f"  ⚠️  [{f['severity']:6s}] "
              f"{f['body_part']:10s} — {f['description']}")
    print(f"{'='*55}\n")

    # ── Confidence chart — shows final decision clearly ───────────
    fig_c, ax_c = plt.subplots(figsize=(10, 5))
    fig_c.patch.set_facecolor('#1a1a2e')
    ax_c.set_facecolor('#16213e')

    # Build display values
    # If biomechanics overrode VideoMAE, show that clearly
    videomae_event = LABEL_TO_EVENT[pred_idx]
    overridden     = (videomae_event != pred_event)

    display_probs  = probs * 100
    bar_colors     = []
    for i in range(NUM_CLASSES):
        if LABEL_TO_EVENT[i] == pred_event:
            bar_colors.append('#e74c3c')   # final prediction = red
        elif LABEL_TO_EVENT[i] == videomae_event and overridden:
            bar_colors.append('#f39c12')   # overridden = orange
        else:
            bar_colors.append('#2c3e70')   # others = dark blue

    bars = ax_c.barh(
        [LABEL_TO_EVENT[i] for i in range(NUM_CLASSES)],
        display_probs,
        color=bar_colors,
        edgecolor='white', linewidth=0.5
    )
    for bar, p in zip(bars, display_probs):
        ax_c.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height()/2,
            f'{p:.1f}%', va='center',
            color='white', fontsize=10
        )

    ax_c.set_xlim(0, 115)
    ax_c.set_xlabel('VideoMAE Confidence %',
                     color='white', fontsize=11)

    # Title shows what actually happened
    if overridden:
        title = (f'Final Decision: {pred_event}  '
                 f'(Biomechanics override — '
                 f'VideoMAE said {videomae_event})')
    else:
        title = (f'Final Decision: {pred_event}  '
                 f'({mae_confidence:.1f}% confidence)')

    ax_c.set_title(title, color='white', fontsize=12,
                    fontweight='bold', pad=12)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Final prediction'),
    ]
    if overridden:
        legend_elements.append(
            Patch(facecolor='#f39c12',
                  label='VideoMAE raw prediction (overridden)')
        )
    legend_elements.append(
        Patch(facecolor='#2c3e70', label='Other classes')
    )
    ax_c.legend(handles=legend_elements,
                loc='lower right',
                facecolor='#1a1a2e',
                labelcolor='white',
                fontsize=9)

    ax_c.tick_params(colors='white')
    for sp in ax_c.spines.values():
        sp.set_edgecolor('#334')
    plt.tight_layout()
    plt.savefig(f'{BASE}/event_classification.png',
                dpi=130, bbox_inches='tight',
                facecolor='#1a1a2e')
    plt.show()

    # Print clear decision summary
    print(f"\n{'='*55}")
    print(f"  DECISION SUMMARY")
    print(f"{'='*55}")
    print(f"  VideoMAE prediction : {videomae_event} "
          f"({probs[pred_idx]*100:.1f}%)")
    if overridden:
        print(f"  Biomechanics override → {pred_event}")
        print(f"  Reason: Sprint biomechanical signature detected")
    else:
        print(f"  Final event         : {pred_event}")
    print(f"{'='*55}\n")

    # Full report
    generate_full_report(
        avg_meas, findings,
        pred_event, event_type,
        mae_confidence, best_frame,
        source_name=filename
    )
    return {
        "event": pred_event,
        "confidence": mae_confidence,
        "risk_score": risk,
        "findings": findings,
        "event_type": event_type,
        "videomae_event": videomae_event
    }


if __name__ == "__main__":
    VIDEO_PATH = "videos/javelinthrow.mp4"

    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found: {VIDEO_PATH}")
    else:
        predict_and_analyse(VIDEO_PATH)
