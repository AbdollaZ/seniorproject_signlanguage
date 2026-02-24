import cv2
import time
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


NUM_CLASSES = 29
MODEL_PATH = "resnet18_28class.pth"
CAMERA_INDEX = 0

CONFIRM_TIME = 3.0
CONF_THRESHOLD = 0.65

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

train_dir = r"D:\Sign_language_dataset\asl_alphabet_train_30"
tmp = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
CLASS_NAMES = tmp.classes
assert len(CLASS_NAMES) == NUM_CLASSES

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

try:
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
except TypeError:
    state = torch.load(MODEL_PATH, map_location=DEVICE)

model.load_state_dict(state)
model.to(DEVICE)
model.eval()


class ASLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Sign Language Recognition")
        self.root.geometry("980x640")
        self.root.minsize(900, 600)

        style = ttk.Style()
        style.theme_use("clam")

        bg = "#0f172a"
        card = "#111c33"
        text = "#e5e7eb"
        muted = "#94a3b8"
        accent = "#38bdf8"
        good = "#22c55e"
        warn = "#f59e0b"
        bad = "#ef4444"

        self.colors = dict(bg=bg, card=card, text=text, muted=muted,
                           accent=accent, good=good, warn=warn, bad=bad)

        self.root.configure(bg=bg)

        style.configure("TFrame", background=bg)
        style.configure("Card.TFrame", background=card)
        style.configure("TLabel", background=bg, foreground=text, font=("Segoe UI", 11))
        style.configure("Title.TLabel", background=bg, foreground=text, font=("Segoe UI", 20, "bold"))
        style.configure("Muted.TLabel", background=bg, foreground=muted, font=("Segoe UI", 11))
        style.configure("Card.TLabel", background=card, foreground=text, font=("Segoe UI", 11))
        style.configure("Big.TLabel", background=card, foreground=text, font=("Segoe UI", 36, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=10)
        style.map("Accent.TButton",
                  background=[("active", "#0ea5e9")],
                  foreground=[("active", "white")])

        style.configure("Conf.Horizontal.TProgressbar",
                        troughcolor="#0b1224",
                        background=accent,
                        thickness=16)

        outer = ttk.Frame(root)
        outer.pack(fill="both", expand=True, padx=18, pady=18)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 12))

        ttk.Label(header, text="ASL Sign Language Recognition", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header,
                  text="Use your right hand only",
                  style="Muted.TLabel").pack(anchor="w", pady=(4, 0))

        main = ttk.Frame(outer)
        main.pack(fill="both", expand=True)

        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        left_card = ttk.Frame(main, style="Card.TFrame")
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        left_card.columnconfigure(0, weight=1)
        left_card.rowconfigure(1, weight=1)

        ttk.Label(left_card, text="Webcam", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=14, pady=(12, 8))

        self.video_label = ttk.Label(left_card)
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        status_card = ttk.Frame(right, style="Card.TFrame")
        status_card.grid(row=0, column=0, sticky="ew")
        status_card.columnconfigure(0, weight=1)

        ttk.Label(status_card, text="Live Prediction", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))

        self.pred_label = ttk.Label(status_card, text="Current: -", style="Card.TLabel")
        self.pred_label.grid(row=1, column=0, sticky="w", padx=14)

        self.conf_bar = ttk.Progressbar(status_card, style="Conf.Horizontal.TProgressbar",
                                        orient="horizontal", length=260, mode="determinate", maximum=100)
        self.conf_bar.grid(row=2, column=0, sticky="ew", padx=14, pady=(8, 12))

        confirmed_card = ttk.Frame(right, style="Card.TFrame")
        confirmed_card.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        confirmed_card.columnconfigure(0, weight=1)
        confirmed_card.rowconfigure(3, weight=1)

        ttk.Label(confirmed_card, text="Confirmed", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))

        self.confirmed_big = ttk.Label(confirmed_card, text="-", style="Big.TLabel")
        self.confirmed_big.grid(row=1, column=0, sticky="w", padx=14)

        self.confirmed_label = ttk.Label(confirmed_card, text="Last confirmed: -", style="Card.TLabel")
        self.confirmed_label.grid(row=2, column=0, sticky="w", padx=14, pady=(0, 10))

        ttk.Label(confirmed_card, text="Output Text", style="Card.TLabel").grid(row=3, column=0, sticky="w", padx=14)

        self.text_box = tk.Text(confirmed_card, height=6, wrap="word",
                                font=("Segoe UI", 16),
                                bg="#0b1224", fg=text, insertbackground=text,
                                relief="flat", padx=12, pady=10)
        self.text_box.grid(row=4, column=0, sticky="nsew", padx=14, pady=(8, 14))
        self.text_box.configure(state="disabled")

        btns = ttk.Frame(outer)
        btns.pack(fill="x", pady=(12, 0))

        self.start_btn = ttk.Button(btns, text="Start Camera", command=self.start)
        self.start_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop)
        self.stop_btn.pack(side="left", padx=(0, 8))

        self.clear_btn = ttk.Button(btns, text="Clear Text", command=self.clear_text)
        self.clear_btn.pack(side="left")

        self.hint = ttk.Label(btns, text=f"Confirm: {CONFIRM_TIME:.1f}s  •  Threshold: {CONF_THRESHOLD:.2f}",
                              style="Muted.TLabel")
        self.hint.pack(side="right")

        self.cap = None
        self.running = False

        self.current_candidate = None
        self.candidate_start_time = time.time()
        self.last_confirmed_sign = None

        self.text_output = ""
        self.last_added_label = None
        self.released = True

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            self.pred_label.config(text="Current: (camera not found)")
            return
        self.running = True
        self.update_frame()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_label.configure(image="")
        self.pred_label.config(text="Current: -")
        self.confirmed_label.config(text="Last confirmed: -")
        self.confirmed_big.config(text="-")
        self.conf_bar["value"] = 0

    def on_close(self):
        self.stop()
        self.root.destroy()

    def clear_text(self):
        self.text_output = ""
        self._set_textbox("")

        self.last_added_label = None
        self.last_confirmed_sign = None
        self.current_candidate = None
        self.released = True

        self.confirmed_label.config(text="Last confirmed: -")
        self.confirmed_big.config(text="-")
        self.pred_label.config(text="Current: -")
        self.conf_bar["value"] = 0

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_frame)
            return

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        x1, y1 = int(0.55 * w), int(0.2 * h)
        x2, y2 = int(0.95 * w), int(0.8 * h)

        roi = frame[y1:y2, x1:x2]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        x = transform(rgb_roi).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        pred_idx = int(pred_idx.item())
        conf = float(conf.item())
        label = CLASS_NAMES[pred_idx]
        label_lower = label.lower()

        self.pred_label.config(text=f"Current: {label} ({conf*100:.1f}%)")
        self.conf_bar["value"] = conf * 100

        now = time.time()

        if conf < CONF_THRESHOLD or label_lower == "nothing":
            self.released = True
            self.current_candidate = None
            self.candidate_start_time = now
            self.last_confirmed_sign = None  #repeat letters
        else:
            if self.current_candidate == pred_idx:
                if now - self.candidate_start_time >= CONFIRM_TIME:
                    if self.last_confirmed_sign != pred_idx:
                        self.confirm_sign(pred_idx)
            else:
                self.current_candidate = pred_idx
                self.candidate_start_time = now

        preview = cv2.resize(rgb_roi, (640, 420), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(preview)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def confirm_sign(self, pred_idx):
        label = CLASS_NAMES[pred_idx]
        label_lower = label.lower()

        if label_lower == "nothing":
            return

        if label == self.last_added_label and not self.released:
            return

        self.last_confirmed_sign = pred_idx
        self.confirmed_label.config(text=f"Last confirmed: {label}")
        self.confirmed_big.config(text=label if len(label) <= 3 else label[:3])

        if label_lower == "space":
            self.text_output += " "
        elif label_lower in ["del", "delete"]:
            self.text_output = self.text_output[:-1]
        else:
            self.text_output += label

        self.last_added_label = label
        self.released = False

        self._set_textbox(self.text_output)

    def _set_textbox(self, text):
        self.text_box.configure(state="normal")
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", text)
        self.text_box.configure(state="disabled")

root = tk.Tk()
app = ASLApp(root)
root.mainloop()
