import cv2
import numpy as np
import torch

from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification
)

# --------------------------------------------------
# Settings
# --------------------------------------------------

VIDEO_PATH = "videos/javelinthrow.mp4"
MODEL_PATH = "model/videomae_trackfield_final.pth"

MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"

LABELS = [
    "Long Jump",
    "High Jump",
    "Pole Vault",
    "Shot Put",
    "Discus Throw",
    "Javelin Throw",
    "Sprint"
]

# --------------------------------------------------
# Extract exactly 16 frames
# Same method as your Kaggle notebook
# --------------------------------------------------

def extract_frames(video_path, num_frames=16):

    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Total video frames:", total)

    if total < num_frames:
        cap.release()
        return None

    indices = np.linspace(
        0,
        total - 1,
        num_frames,
        dtype=int
    )

    frames = []

    for idx in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(idx)
        )

        ret, frame = cap.read()

        if not ret:
            cap.release()
            return None

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(frame)

    cap.release()

    return frames


# --------------------------------------------------
# Load video
# --------------------------------------------------

print("\nExtracting video frames...")

frames = extract_frames(VIDEO_PATH)

if frames is None:
    print("ERROR: Could not extract 16 frames.")
    raise SystemExit

print("Frames extracted:", len(frames))


# --------------------------------------------------
# Load VideoMAE processor
# --------------------------------------------------

print("\nLoading VideoMAE processor...")

processor = VideoMAEImageProcessor.from_pretrained(
    MODEL_NAME
)


# --------------------------------------------------
# Create model
# --------------------------------------------------

print("Creating VideoMAE model...")

model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_NAME,
    num_labels=7,
    ignore_mismatched_sizes=True
)


# --------------------------------------------------
# Load your trained weights
# --------------------------------------------------

print("Loading trained Track & Field weights...")

state = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state)

model.eval()


# --------------------------------------------------
# Prepare video for VideoMAE
# --------------------------------------------------

print("Preparing frames for VideoMAE...")

inputs = processor(
    images=frames,
    return_tensors="pt"
)

pixel_values = inputs["pixel_values"]

print("Input shape:", pixel_values.shape)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

print("\nRunning prediction on CPU...")

with torch.no_grad():

    outputs = model(
        pixel_values=pixel_values
    )

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]


# --------------------------------------------------
# Results
# --------------------------------------------------

predicted_index = int(
    probabilities.argmax()
)

predicted_event = LABELS[predicted_index]

confidence = float(
    probabilities[predicted_index]
) * 100

print("\n" + "=" * 50)
print("VIDEO CLASSIFICATION RESULT")
print("=" * 50)

print(
    f"Prediction : {predicted_event}"
)

print(
    f"Confidence : {confidence:.2f}%"
)

print("\nAll class probabilities:")

for label, probability in zip(
    LABELS,
    probabilities
):

    print(
        f"{label:15s}: "
        f"{float(probability) * 100:.2f}%"
    )

print("=" * 50)