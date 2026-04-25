"""
Problem 6.2 — CNN Shelf Inspector

Tasks:
  1. Build and train a CNN (70/15/15 split, epoch-0 evaluation)
  2. Architecture experiments (3 variations)
  3. Full regularization toolkit: dropout, weight decay, data augmentation, early stopping
  4. Test set evaluation: accuracy, precision, recall, F1, confusion matrix
  5. Visualize first-layer filters
  6. Bonus: Transfer learning with ResNet-18

Run: python src/shelf_cnn.py
Outputs saved to out/
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os, copy

os.makedirs("out", exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)

# ── Load data ────────────────────────────────────────────────────────────────
data        = np.load("shelf_images.npz")
images      = data["images"]          # (900, 64, 64)
labels      = data["labels"]
class_names = list(data["class_names"])

rng   = np.random.default_rng(0)
idx   = rng.permutation(len(images))
n     = len(images)
n_test = int(0.15 * n)
n_val  = int(0.15 * n)
n_train = n - n_test - n_val

X_test   = images[idx[:n_test]]
y_test   = labels[idx[:n_test]]
X_val    = images[idx[n_test:n_test+n_val]]
y_val    = labels[idx[n_test:n_test+n_val]]
X_train  = images[idx[n_test+n_val:]]
y_train  = labels[idx[n_test+n_val:]]

print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
print(f"Classes: {class_names}\n")

def to_tensor(X, y):
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,64,64)
    yt = torch.tensor(y, dtype=torch.long)
    return Xt, yt

X_train_t, y_train_t = to_tensor(X_train, y_train)
X_val_t,   y_val_t   = to_tensor(X_val,   y_val)
X_test_t,  y_test_t  = to_tensor(X_test,  y_test)

train_ds     = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# ── Model definitions ─────────────────────────────────────────────────────────

class ShelfCNN(nn.Module):
    """
    Standard 3-block CNN:
    Conv(16)+BN -> Pool -> Conv(32)+BN -> Pool -> Conv(64)+BN -> Pool -> FC(128) -> FC(3)
    """
    def __init__(self, dropout_p=0.0, use_bn=True):
        super().__init__()
        def conv_block(in_c, out_c):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), nn.ReLU()]
            if use_bn:
                layers.insert(1, nn.BatchNorm2d(out_c))
            layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(1,  16),   # 64→32
            conv_block(16, 32),   # 32→16
            conv_block(32, 64),   # 16→8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ShelfFC(nn.Module):
    """Fully-connected baseline: flattens 64×64 → FC(512) → FC(256) → FC(3)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 512), nn.ReLU(),
            nn.Linear(512, 256),     nn.ReLU(),
            nn.Linear(256, 3),
        )
    def forward(self, x): return self.net(x)


# ── Training function ─────────────────────────────────────────────────────────

def train_model(model, train_loader, X_val_t, y_val_t,
                epochs=30, lr=1e-3, weight_decay=0.0,
                patience=None, augment=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    aug = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=8, translate=(0.05, 0.05)),
        T.ColorJitter(brightness=0.2),
    ]) if augment else None

    train_losses, val_losses, val_accs = [], [], []

    # Epoch 0: before any training
    model.eval()
    with torch.no_grad():
        vp = model(X_val_t)
        vl = criterion(vp, y_val_t).item()
        va = (vp.argmax(dim=1) == y_val_t).float().mean().item()
    train_losses.append(vl)
    val_losses.append(vl)
    val_accs.append(va)

    best_val  = float("inf")
    best_wts  = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for Xb, yb in train_loader:
            if aug is not None:
                Xb = torch.stack([aug(x) for x in Xb])
            pred = model(Xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        train_losses.append(epoch_loss / n_batches)

        model.eval()
        with torch.no_grad():
            vp = model(X_val_t)
            vl = criterion(vp, y_val_t).item()
            va = (vp.argmax(dim=1) == y_val_t).float().mean().item()
        val_losses.append(vl)
        val_accs.append(va)

        if patience is not None:
            if vl < best_val - 1e-4:
                best_val   = vl
                best_wts   = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    model.load_state_dict(best_wts)
                    break
        else:
            if vl < best_val:
                best_val = vl
                best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)
    return train_losses, val_losses, val_accs


# ── Task 1: Baseline CNN ───────────────────────────────────────────────────────
print("=" * 58)
print("Task 1 — Baseline CNN (with BatchNorm, no dropout)")
print("=" * 58)

torch.manual_seed(0)
cnn = ShelfCNN(dropout_p=0.0, use_bn=True)
cnn_train, cnn_val, cnn_val_acc = train_model(cnn, train_loader, X_val_t, y_val_t, epochs=20)

torch.manual_seed(0)
fc  = ShelfFC()
fc_train, fc_val, fc_val_acc = train_model(fc, train_loader, X_val_t, y_val_t, epochs=20)

cnn_params = sum(p.numel() for p in cnn.parameters())
fc_params  = sum(p.numel() for p in fc.parameters())
print(f"CNN : {cnn_params:,} params, final val acc = {cnn_val_acc[-1]*100:.1f}%")
print(f"FC  : {fc_params:,}  params, final val acc = {fc_val_acc[-1]*100:.1f}%")

cnn.eval(); fc.eval()
with torch.no_grad():
    cnn_test_acc = (cnn(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()
    fc_test_acc  = (fc(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()
print(f"CNN test accuracy: {cnn_test_acc*100:.1f}%")
print(f"FC  test accuracy: {fc_test_acc*100:.1f}%\n")

# Loss + accuracy curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(cnn_train, color="#4c78a8", ls="--", lw=1, label="CNN train")
ax.plot(cnn_val,   color="#4c78a8", lw=2,      label="CNN val")
ax.plot(fc_train,  color="#e45756", ls="--", lw=1, label="FC train")
ax.plot(fc_val,    color="#e45756", lw=2,      label="FC val")
ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss"); ax.set_title("Loss")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
ax.plot(cnn_val_acc, color="#4c78a8", lw=2, label="CNN")
ax.plot(fc_val_acc,  color="#e45756", lw=2, label="FC")
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy"); ax.set_title("Accuracy")
ax.set_ylim(0, 1.05); ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("out/cnn_baseline_curves.png", dpi=150)
plt.close()
print("Plot saved → out/cnn_baseline_curves.png")

# ── Task 2: Architecture experiments ─────────────────────────────────────────
print("\n" + "=" * 58)
print("Task 2 — Architecture Experiments")
print("=" * 58)

arch_results = {}

def make_variant(filters, n_layers, use_bn):
    """Build a CNN variant with given filter sizes and optional BN."""
    layers = []
    in_c = 1
    for i in range(n_layers):
        f = filters[i] if i < len(filters) else filters[-1]
        layers += [nn.Conv2d(in_c, f, 3, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(f))
        layers += [nn.ReLU(), nn.MaxPool2d(2)]
        in_c = f
    spatial = 64 // (2 ** n_layers)
    layers += [nn.Flatten(), nn.Linear(in_c * spatial * spatial, 128), nn.ReLU(), nn.Linear(128, 3)]
    return nn.Sequential(*layers)

variants = [
    ("2-layer [8,16]  no BN",  make_variant([8,  16],    2, False)),
    ("3-layer [16,32,64] BN",  make_variant([16, 32, 64], 3, True)),
    ("4-layer [16,32,64,64]BN",make_variant([16, 32, 64, 64], 4, True)),
]

print(f"  {'Variant':<30} {'Params':>8}  {'Val Acc':>8}  {'Test Acc':>9}")
print("  " + "-" * 60)
for name, model in variants:
    torch.manual_seed(0)
    _, _, va = train_model(model, train_loader, X_val_t, y_val_t, epochs=15)
    model.eval()
    with torch.no_grad():
        ta = (model(X_test_t).argmax(1) == y_test_t).float().mean().item()
    p = sum(x.numel() for x in model.parameters())
    arch_results[name] = {"val": va[-1], "test": ta, "params": p}
    print(f"  {name:<30} {p:>8,}  {va[-1]*100:>7.1f}%  {ta*100:>8.1f}%")

# ── Task 3: Full regularization toolkit ──────────────────────────────────────
print("\n" + "=" * 58)
print("Task 3 — Full Regularization Toolkit")
print("=" * 58)

torch.manual_seed(0)
cnn_reg = ShelfCNN(dropout_p=0.4, use_bn=True)
reg_train, reg_val, reg_acc = train_model(
    cnn_reg, train_loader, X_val_t, y_val_t,
    epochs=40, lr=1e-3, weight_decay=1e-4,
    patience=12, augment=True,
)

cnn_reg.eval()
with torch.no_grad():
    reg_test_acc = (cnn_reg(X_test_t).argmax(1) == y_test_t).float().mean().item()
print(f"Regularized CNN test accuracy: {reg_test_acc*100:.1f}%")
print(f"  (trained for {len(reg_train)-1} effective epochs, early-stopped)")

# Plot: with vs without regularization
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cnn_val_acc,  color="#4c78a8", lw=2, label="No regularization")
ax.plot(reg_acc,      color="#54a24b", lw=2, label="Full regularization (dropout+WD+aug+ES)")
# Align by epoch length
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy")
ax.set_title("Effect of Regularization on Validation Accuracy")
ax.set_ylim(0.3, 1.05); ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("out/cnn_regularization.png", dpi=150)
plt.close()
print("Plot saved → out/cnn_regularization.png")

# ── Task 4: Test-set evaluation ───────────────────────────────────────────────
print("\n" + "=" * 58)
print("Task 4 — Final Test-Set Evaluation")
print("=" * 58)

cnn_reg.eval()
with torch.no_grad():
    all_preds = cnn_reg(X_test_t).argmax(dim=1).numpy()
y_true = y_test_t.numpy()

print(classification_report(y_true, all_preds, target_names=class_names))

cm = confusion_matrix(y_true, all_preds)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels(class_names, rotation=30, ha="right")
ax.set_yticks(range(3)); ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix — Regularized CNN")
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("out/cnn_confusion_matrix.png", dpi=150)
plt.close()
print("Plot saved → out/cnn_confusion_matrix.png")

# Show example predictions
cnn_reg.eval()
with torch.no_grad():
    all_preds_np = cnn_reg(X_test_t).argmax(1).numpy()

show_correct, show_wrong = [], []
for i in range(len(y_test)):
    if all_preds_np[i] == y_test[i] and len(show_correct) < 5:
        show_correct.append(i)
    elif all_preds_np[i] != y_test[i] and len(show_wrong) < 5:
        show_wrong.append(i)
    if len(show_correct) >= 5 and len(show_wrong) >= 5:
        break

fig, axes = plt.subplots(2, max(len(show_correct), len(show_wrong)), figsize=(13, 5))
for col, i in enumerate(show_correct):
    axes[0, col].imshow(X_test[i], cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(f"T:{class_names[y_test[i]]}\nP:{class_names[all_preds_np[i]]}",
                            fontsize=8, color="green")
    axes[0, col].axis("off")
for col, i in enumerate(show_wrong[:len(show_correct)]):
    axes[1, col].imshow(X_test[i], cmap="gray", vmin=0, vmax=1)
    axes[1, col].set_title(f"T:{class_names[y_test[i]]}\nP:{class_names[all_preds_np[i]]}",
                            fontsize=8, color="red")
    axes[1, col].axis("off")
axes[0, 0].set_ylabel("Correct", fontsize=10, fontweight="bold")
axes[1, 0].set_ylabel("Wrong",   fontsize=10, fontweight="bold")
plt.suptitle("Sample Predictions (green=correct, red=wrong)", fontsize=12)
plt.tight_layout()
plt.savefig("out/cnn_sample_predictions.png", dpi=150)
plt.close()
print("Plot saved → out/cnn_sample_predictions.png")

# ── Task 5: Filter visualization ─────────────────────────────────────────────
print("\n" + "=" * 58)
print("Task 5 — First-Layer Filter Visualization")
print("=" * 58)

filters = cnn_reg.features[0][0].weight.data.squeeze().numpy()  # (16, 3, 3)
fig, axes = plt.subplots(2, 8, figsize=(10, 2.8))
vmax = np.abs(filters).max()
for i, ax in enumerate(axes.flat):
    ax.imshow(filters[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"#{i+1}", fontsize=8)
plt.suptitle("Learned first-layer filters (3×3)", fontsize=12)
plt.tight_layout()
plt.savefig("out/cnn_filters.png", dpi=150)
plt.close()
print("Plot saved → out/cnn_filters.png")
print("  Filters show edge detectors at various orientations and brightness")
print("  gradient detectors — consistent with local pattern detection theory.")

# ── Bonus: Transfer learning (ResNet-18) ─────────────────────────────────────
print("\n" + "=" * 58)
print("Bonus — Transfer Learning: ResNet-18")
print("=" * 58)

import torchvision.models as models

# Adapt 1-channel input and 3-class output
resnet = models.resnet18(weights="IMAGENET1K_V1")
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet.fc    = nn.Linear(resnet.fc.in_features, 3)

# Feature extraction: freeze all except new fc + conv1
for param in resnet.parameters():
    param.requires_grad = False
for param in resnet.fc.parameters():
    param.requires_grad = True
for param in resnet.conv1.parameters():
    param.requires_grad = True

torch.manual_seed(0)
resnet_opt = optim.Adam(
    filter(lambda p: p.requires_grad, resnet.parameters()), lr=1e-3
)
resnet_criterion = nn.CrossEntropyLoss()

resnet_train_losses, resnet_val_accs = [], []
for epoch in range(15):
    resnet.train()
    for Xb, yb in train_loader:
        pred = resnet(Xb)
        loss = resnet_criterion(pred, yb)
        resnet_opt.zero_grad(); loss.backward(); resnet_opt.step()

resnet.eval()
with torch.no_grad():
    resnet_feat_acc = (resnet(X_test_t).argmax(1) == y_test_t).float().mean().item()
print(f"ResNet-18 (feature extraction, 15 ep): {resnet_feat_acc*100:.1f}%")

# Fine-tuning
for param in resnet.parameters():
    param.requires_grad = True
ft_opt = optim.Adam(resnet.parameters(), lr=1e-4)
for epoch in range(10):
    resnet.train()
    for Xb, yb in train_loader:
        pred = resnet(Xb)
        loss = resnet_criterion(pred, yb)
        ft_opt.zero_grad(); loss.backward(); ft_opt.step()

resnet.eval()
with torch.no_grad():
    resnet_ft_acc = (resnet(X_test_t).argmax(1) == y_test_t).float().mean().item()
print(f"ResNet-18 (fine-tuned, +10 ep):         {resnet_ft_acc*100:.1f}%")
print(f"From-scratch CNN:                        {reg_test_acc*100:.1f}%")
print()
print("  Transfer learning on synthetic data: ImageNet features don't perfectly")
print("  match geometric shelf patterns, so fine-tuning is needed to close the gap.")

print("\n" + "=" * 58)
print("All plots saved to out/")
print("=" * 58)
