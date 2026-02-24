from pathlib import Path
from collections import deque
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import VivitImageProcessor, VivitForVideoClassification

MODEL_DIR = Path(r"D:\Projects_Python\Senior\video\vivit_finetuned_first10_ft")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRED_EVERY = 8

FPS_CAP = 20


def load_labels():
    with open(MODEL_DIR / "labels.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def sample_indices(t: int, num_frames: int):
    if t >= num_frames:
        return np.linspace(0, t - 1, num_frames).astype(int)
    return (np.arange(num_frames) % t).astype(int)


@torch.no_grad()
def predict_buffer(frames_bgr, processor, model, labels):
    model.eval()
    target_size = int(getattr(model.config, "image_size", 224))
    num_frames = int(getattr(model.config, "num_frames", 32))

    frames_list = list(frames_bgr)
    idx = sample_indices(len(frames_list), num_frames)
    sampled = [frames_list[i] for i in idx]

    frames_rgb = []
    for f in sampled:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        frames_rgb.append(rgb)

    inputs = processor(frames_rgb, return_tensors="pt", do_resize=False, do_center_crop=False)
    x = inputs["pixel_values"].to(DEVICE)  # (1,T,3,H,W)

    logits = model(pixel_values=x).logits
    probs = F.softmax(logits, dim=1).squeeze(0)

    k = min(3, len(labels))
    topv, topi = torch.topk(probs, k=k)
    return [(labels[i], float(v)) for v, i in zip(topv.cpu(), topi.cpu())]


def main():
    labels = load_labels()
    processor = VivitImageProcessor.from_pretrained(str(MODEL_DIR))
    model = VivitForVideoClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE)

    print("Loaded:", MODEL_DIR)
    print("Device:", DEVICE)
    print("Controls: Q = quit, SPACE = pause/resume\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try VideoCapture(1).")

    num_frames = int(getattr(model.config, "num_frames", 32))
    buffer = deque(maxlen=num_frames)

    last_pred = []
    last_infer_ms = None
    frame_count = 0
    paused = False

    last_time = time.time()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            buffer.append(frame.copy())
            frame_count += 1

            if len(buffer) == buffer.maxlen and (frame_count % PRED_EVERY == 0):
                t0 = time.time()
                try:
                    last_pred = predict_buffer(buffer, processor, model, labels)
                except Exception as e:
                    last_pred = [("ERROR", 0.0)]
                    print("Prediction error:", e)
                last_infer_ms = (time.time() - t0) * 1000.0

        disp = frame if not paused else frame  # keep last frame on pause
        cv2.putText(disp, f"Buffer: {len(buffer)}/{buffer.maxlen}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if last_infer_ms is not None:
            cv2.putText(disp, f"Infer: {last_infer_ms:.0f} ms (every {PRED_EVERY} frames)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        y0 = 95
        if last_pred:
            cv2.putText(disp, "Top-3:", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            for j, (lab, p) in enumerate(last_pred):
                cv2.putText(disp, f"{j+1}. {lab}  {p*100:.1f}%",
                            (10, y0 + 30*(j+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if paused:
            cv2.putText(disp, "PAUSED (press SPACE)", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Real-time ViViT (sliding window)", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        if key == 32:  #space button
            paused = not paused

        if FPS_CAP is not None and not paused:
            now = time.time()
            dt = now - last_time
            target_dt = 1.0 / FPS_CAP
            if dt < target_dt:
                time.sleep(target_dt - dt)
            last_time = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
