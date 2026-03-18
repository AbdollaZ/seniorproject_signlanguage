import torch
import torch.nn as nn
import pathlib
from transformers import VivitImageProcessor, VivitForVideoClassification
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random
import av
from importlib.resources import path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time


dataset_root_path = pathlib.Path("/kaggle/input/datasets/ajlnkk/wlasl-100/wlasl100")

video_count_train = len(list(dataset_root_path.glob("train/*/*.mp4")))
video_count_val = len(list(dataset_root_path.glob("val/*/*.mp4")))
video_count_test = len(list(dataset_root_path.glob("test/*/*.mp4")))
video_total = video_count_train + video_count_val + video_count_test
print(f"Total videos: {video_total}")

all_video_file_paths = (
    list(dataset_root_path.glob("train/*/*.mp4"))
    + list(dataset_root_path.glob("val/*/*.mp4"))
    + list(dataset_root_path.glob("test/*/*.mp4"))
)
all_video_file_paths[:5]

class_labels = sorted({path.parent.name for path in all_video_file_paths})

label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}

print(f"Unique classes ({len(class_labels)}): {class_labels}")


image_processor = VivitImageProcessor.from_pretrained("Shawon16/ViViT_wlasl_100_200ep_coR_")
model = VivitForVideoClassification.from_pretrained(
    "Shawon16/ViViT_wlasl_100_200ep_coR_",
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

DEVICE = torch.device("cuda")
model.to(DEVICE)
print(next(model.parameters()).device)



class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, image_processor, num_frames):
        self.video_paths = video_paths
        self.labels = labels
        self.image_processor = image_processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def _load_video(self, path):
        container = av.open(str(path))
        frames = []

        for frame in container.decode(video=0):
            img = Image.fromarray(frame.to_rgb().to_ndarray())
            img = img.resize((192, 192))
            frames.append(np.array(img))

        container.close()
        return frames

    def _sample_frames(self, frames):
        total = len(frames)

        if total >= self.num_frames:
            step = total // self.num_frames
            idx = [i * step for i in range(self.num_frames)]
        else:
            idx = np.linspace(0, total - 1, self.num_frames).astype(int)

        return [frames[i] for i in idx]

    def __getitem__(self, idx):
        frames = self._load_video(self.video_paths[idx])
        frames = self._sample_frames(frames)

        inputs = self.image_processor(
            frames,
            return_tensors="pt"
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(self.labels[idx])

        return inputs
    


train_paths = list((dataset_root_path / "train").rglob("*.mp4"))
val_paths = list((dataset_root_path / "val").rglob("*.mp4"))
test_paths = list((dataset_root_path / "test").rglob("*.mp4"))

train_labels = [label2id[p.parent.name] for p in train_paths]
val_labels = [label2id[p.parent.name] for p in val_paths]
test_labels = [label2id[p.parent.name] for p in test_paths]

num_frames = model.config.num_frames

train_dataset = VideoDataset(
    train_paths,
    train_labels,
    image_processor,
    num_frames,
    train=True,
)

val_dataset = VideoDataset(
    val_paths,
    val_labels,
    image_processor,
    num_frames,
    train=False,
)

test_dataset = VideoDataset(
    test_paths,
    test_labels,
    image_processor,
    num_frames,
    train=False,
)


train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=2)




EPOCHS = 30
LR = 2e-5
WEIGHT_DECAY = 0.01
GRAD_ACCUM_STEPS = 8
PATIENCE = 5
MIN_INC_ACC = 0.001

SAVE_PATH = "/kaggle/working/vivit-finetuned.pth"



criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scaler = torch.cuda.amp.GradScaler()

best_val_acc = 0
patience_counter = 0

start = time.time()
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 30)

    model.train()
    train_loss = 0
    optimizer.zero_grad()

    train_preds = []
    train_labels = []

    for step, batch in enumerate(tqdm(train_loader)):
        inputs = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item() * GRAD_ACCUM_STEPS

        preds = torch.argmax(logits, dim=1)
        train_preds.extend(preds.detach().cpu().numpy())
        train_labels.extend(labels.detach().cpu().numpy())

    train_acc = accuracy_score(train_labels, train_preds)
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                logits = outputs.logits

            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)

    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Train Acc : {train_acc:.4f}")
    print(f"Val Acc   : {val_acc:.4f}")

    if val_acc > best_val_acc + MIN_INC_ACC:
        print("Validation accuracy improved. Saving model...")
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), SAVE_PATH)
    else:
        patience_counter += 1
        print(f"No significant improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

print("Training complete.")
print(f"Training time: {time.time() - start:.4f}")

model.load_state_dict(torch.load(SAVE_PATH))
model.eval()