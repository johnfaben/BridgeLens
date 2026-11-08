"""Train (by default, fine-tune) the corner classifier on an exported dataset.

Reads <dataset>/classifier/{train,val} and <dataset>/dataset.json from
export_dataset.py. Writes to <out>/:
    classifier.pt   best checkpoint by val accuracy (same format the app loads,
                    plus crop_expand, which inference uses when cropping)
    run.json        settings, dataset info and per-epoch history

Runs locally (CPU, slow) or on a RunPod GPU. Only needs torch + torchvision.

Usage:
    python -m training.train_classifier --dataset training_data/datasets/2026-11-10 --out training_data/runs/cls-1
    python -m training.train_classifier --dataset ... --out ... --init none --epochs 80     # from scratch
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_classifier_cnn import CardClassifier  # noqa: E402

LIVE_MODEL = os.path.join(ROOT, 'best_corner_classifier_cnn.pt')
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_transforms(img_size):
    """Train: augmentation for fanned and angled hands. Val: the inference transform."""
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(25),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02),
        transforms.RandomGrayscale(p=0.15),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        NORMALIZE,
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        NORMALIZE,
    ])
    return train_tf, val_tf


class CropFolder(Dataset):
    """Like ImageFolder, but with a fixed class list (empty classes allowed)."""

    def __init__(self, root, classes, transform):
        self.transform = transform
        self.samples = []
        for idx, cls in enumerate(classes):
            d = os.path.join(root, cls)
            if os.path.isdir(d):
                self.samples += [(os.path.join(d, f), idx) for f in sorted(os.listdir(d))
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert('RGB')), label


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0
    per_class = {}
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            pred = outputs.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            if not training:
                for l, p in zip(labels.tolist(), pred.tolist()):
                    c = per_class.setdefault(l, [0, 0])
                    c[0] += l == p
                    c[1] += 1
    return total_loss / max(1, total), correct / max(1, total), per_class


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', required=True, help='Exported dataset dir (has dataset.json)')
    parser.add_argument('--out', required=True, help='Run output dir')
    parser.add_argument('--init', default=LIVE_MODEL, help="Checkpoint to fine-tune from, or 'none'")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=None, help='Default 3e-4 fine-tuning, 1e-3 from scratch')
    parser.add_argument('--img-size', type=int, default=None, help='Default: from --init, else 64')
    parser.add_argument('--real-weight', type=float, default=3.0,
                        help='Sampling weight of new real-photo crops relative to legacy crops')
    parser.add_argument('--workers', type=int, default=0 if os.name == 'nt' else 4)
    args = parser.parse_args()

    with open(os.path.join(args.dataset, 'dataset.json')) as f:
        info = json.load(f)
    classes = info['classes']
    crop_expand = info['settings']['crop_expand']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init = None if args.init == 'none' else torch.load(args.init, map_location='cpu', weights_only=False)
    if init is not None and list(init['class_names']) != classes:
        sys.exit('Class list in --init does not match the dataset')
    img_size = args.img_size or (init.get('img_size', 64) if init else 64)
    if init is not None and img_size != init.get('img_size', 64):
        print(f'Note: img size {img_size} differs from init ({init.get("img_size", 64)}); weights still load')
    lr = args.lr or (3e-4 if init else 1e-3)

    train_tf, val_tf = get_transforms(img_size)
    train_ds = CropFolder(os.path.join(args.dataset, 'classifier', 'train'), classes, train_tf)
    val_ds = CropFolder(os.path.join(args.dataset, 'classifier', 'val'), classes, val_tf)
    weights = [1.0 if os.path.basename(p).startswith('legacy_') else args.real_weight
               for p, _ in train_ds.samples]
    n_real = sum(w != 1.0 for w in weights)
    print(f'Device {device}; train {len(train_ds)} ({n_real} real-photo, x{args.real_weight}), '
          f'val {len(val_ds)}; img {img_size}, crop_expand {crop_expand}, lr {lr}', flush=True)

    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = CardClassifier(num_classes=len(classes))
    if init is not None:
        model.load_state_dict(init['model_state_dict'])
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, 'classifier.pt')
    run = {'kind': 'classifier', 'dataset': info['name'], 'dataset_settings': info['settings'],
           'args': vars(args), 'img_size': img_size, 'crop_expand': crop_expand, 'lr': lr,
           'train_size': len(train_ds), 'val_size': len(val_ds), 'history': []}

    _, init_acc, _ = run_epoch(model, val_loader, criterion, device)
    print(f'Val acc before training: {init_acc:.4f}', flush=True)
    run['val_acc_init'] = init_acc
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        vl, va, per_class = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        run['history'].append({'epoch': epoch, 'train_loss': tl, 'train_acc': ta,
                               'val_loss': vl, 'val_acc': va, 'secs': round(time.time() - t0, 1)})
        print(f'Epoch {epoch:3d}/{args.epochs} train {tl:.4f}/{ta:.3f} val {vl:.4f}/{va:.4f} '
              f'({time.time() - t0:.0f}s)', flush=True)
        if va > best:
            best = va
            worst = sorted(((c[0] / c[1], classes[k], c[1]) for k, c in per_class.items()))[:8]
            torch.save({'model_state_dict': model.state_dict(), 'class_names': classes,
                        'num_classes': len(classes), 'img_size': img_size, 'crop_expand': crop_expand,
                        'val_acc': va, 'epoch': epoch, 'dataset': info['name'],
                        'init': args.init}, ckpt_path)
            run.update(best_epoch=epoch, best_val_acc=va,
                       worst_val_classes=[[n, round(a, 3), cnt] for a, n, cnt in worst])
        with open(os.path.join(args.out, 'run.json'), 'w') as f:
            json.dump(run, f, indent=2)

    print(f'Best val acc {best:.4f} (was {init_acc:.4f} before training) -> {ckpt_path}')


if __name__ == '__main__':
    main()
