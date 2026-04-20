import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import os

NUM_CLASSES = 29
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

train_dir = r"D:\Sign_language_dataset\asl_alphabet_train_300"
test_dir  = r"D:\Sign_language_dataset\asl_alphabet_test"

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class ResNetLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers):
        super(ResNetLSTM, self).__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.input_size = 512
        self.seq_len = 7 * 7
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x) 
        x = x.permute(0, 2, 3, 1) 
        x = x.reshape(x.size(0), -1, 512)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = ResNetLSTM(num_classes=NUM_CLASSES,
                   hidden_size=HIDDEN_SIZE,
                   num_layers=NUM_LAYERS).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

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

torch.save(model.state_dict(), "resnet18_lstm_28class.pth")
print("Model saved to resnet18_lstm_28class.pth")
