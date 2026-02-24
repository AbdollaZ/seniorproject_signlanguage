import sys
import os
import types
import torchvision

if not hasattr(torchvision.transforms, 'functional_tensor'):
    import torchvision.transforms.functional as F

    mock_ft = types.ModuleType("functional_tensor")
    mock_ft.rgb_to_grayscale = F.to_grayscale
    mock_ft.to_tensor = F.to_tensor
    sys.modules["torchvision.transforms.functional_tensor"] = mock_ft

import time
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, CenterCrop
from pytorchvideo.transforms import ApplyTransformToKey, Normalize, ShortSideScale, UniformTemporalSubsample
from pytorchvideo.models.hub import slowfast_r50
from peft import LoraConfig, get_peft_model
import warnings
import gc

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")


class PackPathway(torch.nn.Module):
    def __init__(self, alpha=4):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        num_frames = frames.shape[1]
        slow_frames = max(1, num_frames // self.alpha)
        indices = torch.linspace(0, num_frames - 1, slow_frames, device=frames.device).long()
        slow_pathway = torch.index_select(frames, 1, indices)
        return [slow_pathway, fast_pathway]


class VideoTransform:
    def __init__(self, is_training=False):
        mean, std = [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]

        transform = Compose([
            UniformTemporalSubsample(32),
            ShortSideScale(256),
            CenterCrop(224),
            Normalize(mean, std),
        ])
        self.pack_pathway = PackPathway(alpha=4)
        self.apply_transform = ApplyTransformToKey(key="video", transform=transform)

    def __call__(self, x):
        x = self.apply_transform(x)
        x["video"] = self.pack_pathway(x["video"])
        return x


class WLASLDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.samples = []
        self.invalid_samples = set()

        for idx in range(len(self.df)):
            try:
                path = str(self.df.iloc[idx]['path']).strip()
                label = int(self.df.iloc[idx]['label'])

                if os.path.exists(path):
                    self.samples.append((path, label))
            except Exception as e:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for attempt in range(3):
            try:
                path, label = self.samples[idx]

                if idx in self.invalid_samples:
                    idx = (idx + 1) % len(self.samples)
                    continue

                vframes, _, _ = torchvision.io.read_video(path, pts_unit='sec')

                if vframes.size(0) < 8:
                    self.invalid_samples.add(idx)
                    idx = (idx + 1) % len(self.samples)
                    continue

                if vframes.size(0) > 32:
                    vframes = vframes[:32]
                elif vframes.size(0) < 32:
                    pad = torch.zeros(32 - vframes.size(0), *vframes.shape[1:])
                    vframes = torch.cat([vframes, pad], dim=0)

                video = vframes.permute(3, 0, 1, 2).float() / 255.0

                if self.transform:
                    video = self.transform({"video": video})["video"]

                return {"video": video, "label": label}

            except Exception as e:
                self.invalid_samples.add(idx)
                idx = (idx + 1) % len(self.samples)
                continue

        dummy_video = torch.zeros(3, 32, 224, 224)
        if self.transform:
            dummy_video = self.transform({"video": dummy_video})["video"]
        return {"video": dummy_video, "label": 0}


def get_loader(csv_path, transform, batch_size, num_workers, shuffle):
    dataset = WLASLDataset(csv_path, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        prefetch_factor=2 if num_workers > 0 else None
    )


def validate_model(model, loader, criterion, device):
    model.eval()
    correct, total, total_loss, batches = 0, 0, 0.0, 0

    with torch.no_grad():
        for data in loader:
            try:
                inputs = [x.to(device, non_blocking=True) for x in data["video"]]
                labels = data["label"].to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                batches += 1
            except Exception:
                continue

    avg_loss = total_loss / max(batches, 1)
    accuracy = 100 * correct / max(total, 1)
    return avg_loss, accuracy


def main():
    ROOT_DIR = r"C:\Users\Acer\PythonProject\WLASL_100"

    BATCH_SIZE = 8
    NUM_WORKERS = 2
    LR = 1e-4
    EPOCHS = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_csv = os.path.join(ROOT_DIR, "train.csv")
    val_csv = os.path.join(ROOT_DIR, "val.csv")
    test_csv = os.path.join(ROOT_DIR, "test.csv")

    if not os.path.exists(train_csv):
        print(f"ERROR: {train_csv} not found!")
        return

    train_transform = VideoTransform(is_training=True)
    val_transform = VideoTransform(is_training=False)

    train_loader = get_loader(train_csv, train_transform, BATCH_SIZE, NUM_WORKERS, True)
    val_loader = get_loader(val_csv, val_transform, BATCH_SIZE, NUM_WORKERS, False)
    test_loader = get_loader(test_csv, val_transform, BATCH_SIZE, NUM_WORKERS, False)

    if len(train_loader.dataset) == 0:
        print("ERROR: Training dataset is empty!")
        return

    model = slowfast_r50(pretrained=True)
    model.blocks[6].proj = nn.Linear(model.blocks[6].proj.in_features, 100)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["proj", "conv1", "conv2", "conv3"],
        lora_dropout=0.05,
        bias="none"
    )

    try:
        model = get_peft_model(model, lora_config)
    except Exception as e:
        pass

    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if DEVICE == "cuda" else None

    print("Starting Fine-Tuning on WLASL-100...")
    print(f"{'Epoch':<6} | {'Time':<8} | {'Tr. Loss':<9} | {'Val Loss':<9} | {'Val Acc':<8} | {'Test Acc':<8}")
    print("-" * 75)


    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        train_loss, train_batches = 0.0, 0

        for batch_idx, data in enumerate(train_loader):
            try:
                inputs = [x.to(DEVICE, non_blocking=True) for x in data["video"]]
                labels = data["label"].to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if DEVICE == "cuda" and scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            except Exception as e:
                continue

        epoch_time = time.time() - start_time
        avg_train_loss = train_loss / max(train_batches, 1)

        v_loss, v_acc = validate_model(model, val_loader, criterion, DEVICE)
        t_loss, t_acc = validate_model(model, test_loader, criterion, DEVICE)

        scheduler.step()

        print(f"{epoch + 1:02d}/{EPOCHS:02d} | {epoch_time:6.2f}s | {avg_train_loss:8.4f} | "
              f"{v_loss:8.4f} | {v_acc:6.2f}% | {t_acc:6.2f}%")


        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print("\nTraining Finished for WLASL-100.")


if __name__ == '__main__':
    main()