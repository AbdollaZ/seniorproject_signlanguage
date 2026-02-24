import sys
import os
import types
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

if not hasattr(torchvision.transforms, 'functional_tensor'):
    import torchvision.transforms.functional as F

    mock_ft = types.ModuleType("functional_tensor")
    mock_ft.rgb_to_grayscale = F.to_grayscale
    mock_ft.to_tensor = F.to_tensor
    sys.modules["torchvision.transforms.functional_tensor"] = mock_ft

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
    ShortSideScale
)
from torchvision.transforms import Compose, CenterCrop, RandomCrop, RandomHorizontalFlip
from pytorchvideo.models.hub import slowfast_r50
from peft import LoraConfig, get_peft_model


class PackPathway(torch.nn.Module):
    def __init__(self, alpha=4):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        num_frames = frames.shape[1]
        indices = torch.linspace(0, num_frames - 1, num_frames // self.alpha).long().to(frames.device)
        slow_pathway = torch.index_select(frames, 1, indices)
        return [slow_pathway, fast_pathway]


def get_transform(is_training=True):
    mean, std = [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
    transforms = [
        UniformTemporalSubsample(32),
        Normalize(mean, std),
    ]
    if is_training:
        transforms += [RandomShortSideScale(min_size=256, max_size=320), RandomCrop(224), RandomHorizontalFlip(p=0.5)]
    else:
        transforms += [ShortSideScale(256), CenterCrop(224)]

    transforms.append(PackPathway(alpha=4))
    return ApplyTransformToKey(key="video", transform=Compose(transforms))


class KineticsFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir): continue
            for vid_name in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, vid_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        vframes, _, _ = torchvision.io.read_video(path, pts_unit='sec')
        video = vframes.permute(3, 0, 1, 2).float() / 255.0
        if self.transform:
            video = self.transform({"video": video})["video"]
        return {"video": video, "label": label}


def main():
    ROOT_TRAIN_DIR = r"C:\Users\Acer\PythonProject\kinetics600_5per\train"
    BATCH_SIZE, EPOCHS, LR = 8, 10, 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    full_dataset = KineticsFolderDataset(ROOT_TRAIN_DIR)
    num_classes = len(full_dataset.classes)

    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_dataset = Subset(full_dataset, train_idx)
    train_dataset.dataset.transform = get_transform(is_training=True)

    val_dataset = Subset(full_dataset, val_idx)
    val_dataset.dataset.transform = get_transform(is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = slowfast_r50(pretrained=True)
    model.blocks[6].proj = nn.Linear(model.blocks[6].proj.in_features, num_classes)

    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["conv", "proj"],
        lora_dropout=0.05, bias="none"
    )
    model = get_peft_model(model, lora_config).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))

    print(f"\nTraining on {num_classes} classes from 5% subset...")
    print(f"{'Epoch':<8} | {'Tr. Loss':<10} | {'Val Acc':<8}")
    print("-" * 40)

    for epoch in range(EPOCHS):
        model.train()
        train_loss, batches = 0.0, 0
        for data in train_loader:
            try:
                inputs = [t.to(DEVICE) for t in data["video"]]
                labels = data["label"].to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                batches += 1
            except Exception:
                continue

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data in val_loader:
                try:
                    inputs = [t.to(DEVICE) for t in data["video"]]
                    labels = data["label"].to(DEVICE)
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()
                except Exception:
                    continue

        avg_loss = train_loss / batches if batches > 0 else 0
        accuracy = (100 * correct / total) if total > 0 else 0
        print(f"{epoch + 1:02d}/{EPOCHS:02d} | {avg_loss:8.4f} | {accuracy:6.2f}%")

if __name__ == '__main__':
    main()