"""Train a lightweight CNN classifier for card corner crops.

Much smaller/faster than YOLO-cls — just a few conv layers for 53-class
(52 cards + XX) symbol recognition on ~75x75 crops.

Usage:
    python train_classifier_cnn.py
    python train_classifier_cnn.py --epochs 100 --batch-size 64
    python train_classifier_cnn.py --resume best_corner_classifier_cnn.pt
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "classifier_data"
MODEL_PATH = ROOT / "best_corner_classifier_cnn.pt"


def get_transforms(img_size=64):
    """Training transforms with augmentation, val transforms without."""
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02),
        transforms.RandomGrayscale(p=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


class CardClassifier(nn.Module):
    """Small CNN for card corner classification.

    Architecture: 4 conv blocks -> global avg pool -> FC head.
    ~200K parameters (vs ~5M for YOLO11s-cls).
    """
    def __init__(self, num_classes=53):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64x64x3 -> 32x32x32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32x32x32 -> 16x16x64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 16x16x64 -> 8x8x128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 8x8x128 -> 4x4x128
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description='Train card corner CNN classifier')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    train_tf, val_tf = get_transforms(args.img_size)
    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=val_tf)

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    print(f"Classes: {num_classes} ({', '.join(class_names[:5])}...)")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Model
    model = CardClassifier(num_classes=num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from {args.resume}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  lr={lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'num_classes': num_classes,
                'img_size': args.img_size,
                'val_acc': val_acc,
                'epoch': epoch + 1,
            }, MODEL_PATH)
            print(f"  -> Saved best model ({val_acc:.3f})")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to {MODEL_PATH}")

    # Save class names mapping for inference
    class_map_path = ROOT / "classifier_classes.json"
    json.dump({i: name for i, name in enumerate(class_names)},
              open(class_map_path, 'w'))
    print(f"Class mapping saved to {class_map_path}")


if __name__ == '__main__':
    main()
