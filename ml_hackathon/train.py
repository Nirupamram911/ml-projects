import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import FireDataset
from model import SimpleCNN

# Paths
IMG_DIR = "data/images"
MASK_DIR = "data/masks"

# Dataset
dataset = FireDataset(IMG_DIR, MASK_DIR)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model
model = SimpleCNN().to(device)

# Loss + Optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 🔁 Training loop
epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 💾 Save model
torch.save(model.state_dict(), "model.pth")

# 👀 Visualize prediction
model.eval()

x, y = dataset[0]

with torch.no_grad():
    pred = model(x.unsqueeze(0).to(device))[0][0].cpu().numpy()

plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.title("Input")
plt.imshow(x.permute(1,2,0))

plt.subplot(1,3,2)
plt.title("Ground Truth")
plt.imshow(y[0], cmap="hot")

plt.subplot(1,3,3)
plt.title("Prediction")
plt.imshow(pred, cmap="hot")

plt.show()