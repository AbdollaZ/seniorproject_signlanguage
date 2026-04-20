import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os

# --- PARAMETERS ---
IMG_SIZE = 300
NUM_CLASSES = 29
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("WARNING: Training will run on CPU")
# --- DATA PREPARATION ---
# Expected structure:
# data/
#   train/
#     class1/
#     class2/
#     ...
#   test/
#     class1/
#     ...

train_dir = r"D:\Sign_language_dataset\asl_alphabet_train_300"
test_dir  = r"D:\Sign_language_dataset\asl_alphabet_test"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # shape: [3, 28, 28], values in [0,1]
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# --- MODEL ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # input_size = 3 * IMG_SIZE (each row flattened)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, 3, 28, 28]
        # reshape to [batch, seq_len=28, input_size=3*28]
        x = x.permute(0, 2, 3, 1)  # [batch, 28, 28, 3]
        x = x.reshape(x.size(0), IMG_SIZE, -1)  # [batch, 28, 84]

        out, _ = self.lstm(x)        # out: [batch, seq_len, hidden_size]
        out = out[:, -1, :]          # take the last time step
        out = self.fc(out)           # [batch, num_classes]
        return out

# --- INSTANTIATE MODEL ---
input_size = 3 * IMG_SIZE  # 3 channels × 28 pixels per row = 84
model = LSTMClassifier(input_size, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- TRAINING LOOP ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# --- TESTING LOOP ---
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
