"""Train (by default, fine-tune) the corner detector on exported tiles.

Training images = the new tiles in <dataset>/detector/train (listed --new-repeat
times, to weight real photos over the much larger legacy set) plus, if given,
the legacy bridge_ocr corner_detector train set. Validation = the new val tiles
(real photos, split by deal), optionally plus the legacy valid set.

Writes to <out>/:
    detector.pt                         best weights
    best_corner_detector_openvino_model/ OpenVINO export (what the app serves)
    run.json                            settings and metrics
    yolo/                               ultralytics run dir (plots, results.csv)

Needs ultralytics. Meant for a GPU (RunPod); CPU works for a smoke test only.

Usage:
    python -m training.train_detector --dataset /workspace/datasets/2026-11-10 \\
        --legacy /workspace/legacy/corner_detector --out /workspace/runs/det-1
"""

import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_MODEL = os.path.join(ROOT, 'best_corner_detector.pt')
IMG_EXTS = ('.jpg', '.jpeg', '.png')


def list_images(images_dir):
    if not os.path.isdir(images_dir):
        return []
    return sorted(os.path.abspath(os.path.join(images_dir, f)) for f in os.listdir(images_dir)
                  if f.lower().endswith(IMG_EXTS))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', required=True, help='Exported dataset dir (has detector/)')
    parser.add_argument('--legacy', help='Legacy corner_detector dir (train/images, valid/images)')
    parser.add_argument('--legacy-val', action='store_true', help='Also validate on the legacy valid set')
    parser.add_argument('--out', required=True)
    parser.add_argument('--init', default=LIVE_MODEL, help="Weights to fine-tune from, or 'yolov8n.pt' for a fresh start")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--new-repeat', type=int, default=4, help='Times each new tile is listed in train')
    parser.add_argument('--batch', type=int, default=-1, help='-1 = auto for the GPU')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', default=None, help="e.g. 0 or 'cpu' (default: auto)")
    parser.add_argument('--no-export', action='store_true')
    args = parser.parse_args()

    from ultralytics import YOLO

    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    new_train = list_images(os.path.join(args.dataset, 'detector', 'train', 'images'))
    new_val = list_images(os.path.join(args.dataset, 'detector', 'val', 'images'))
    legacy_train = list_images(os.path.join(args.legacy, 'train', 'images')) if args.legacy else []
    legacy_val = list_images(os.path.join(args.legacy, 'valid', 'images')) if args.legacy and args.legacy_val else []
    if not new_train and not legacy_train:
        sys.exit('No training images found')

    train_txt = os.path.join(args.out, 'train.txt')
    val_txt = os.path.join(args.out, 'val.txt')
    with open(train_txt, 'w') as f:
        f.write('\n'.join(legacy_train + new_train * args.new_repeat) + '\n')
    with open(val_txt, 'w') as f:
        f.write('\n'.join(new_val + legacy_val) + '\n')
    data_yaml = os.path.join(args.out, 'data.yaml')
    with open(data_yaml, 'w') as f:
        # JSON strings are valid YAML and survive spaces and Windows backslashes
        f.write(f'train: {json.dumps(train_txt)}\nval: {json.dumps(val_txt)}\nnc: 1\nnames: [card_corner]\n')
    print(f'Train: {len(legacy_train)} legacy + {len(new_train)} new x{args.new_repeat}; '
          f'val: {len(new_val)} new + {len(legacy_val)} legacy', flush=True)

    run = {'kind': 'detector', 'args': vars(args), 'n_legacy_train': len(legacy_train),
           'n_new_train': len(new_train), 'n_new_val': len(new_val), 'n_legacy_val': len(legacy_val)}
    with open(os.path.join(args.dataset, 'dataset.json')) as f:
        run['dataset'] = json.load(f)['name']

    model = YOLO(args.init)
    run['val_init'] = {k: float(v) for k, v in model.val(data=data_yaml, imgsz=640, batch=16,
                                                        device=args.device, plots=False,
                                                        project=os.path.join(args.out, 'yolo'),
                                                        name='val_init').results_dict.items()}
    print('Before training:', run['val_init'], flush=True)

    # Augmentation as for the original detector: no vertical flips (corner symbols
    # must stay readable), small rotations, some perspective for angled photos
    results = model.train(
        data=data_yaml, imgsz=640, epochs=args.epochs, batch=args.batch, workers=args.workers,
        patience=args.patience, device=args.device, mosaic=0.5, degrees=15.0, scale=0.3,
        perspective=0.0005, flipud=0.0, fliplr=0.5,
        project=os.path.join(args.out, 'yolo'), name='train', exist_ok=True, plots=True,
    )
    best = os.path.join(str(results.save_dir), 'weights', 'best.pt')
    shutil.copy2(best, os.path.join(args.out, 'detector.pt'))

    best_model = YOLO(os.path.join(args.out, 'detector.pt'))
    run['val_best'] = {k: float(v) for k, v in best_model.val(data=data_yaml, imgsz=640, batch=16,
                                                             device=args.device, plots=False,
                                                             project=os.path.join(args.out, 'yolo'),
                                                             name='val_best').results_dict.items()}
    print('After training:', run['val_best'], flush=True)
    key = 'metrics/mAP50-95(B)'
    run['improved'] = run['val_best'].get(key, 0) > run['val_init'].get(key, 0)
    if not run['improved']:
        print(f'WARNING: {key} did not improve on the starting weights', flush=True)

    if not args.no_export:
        exported = best_model.export(format='openvino', imgsz=640, half=False)
        dst = os.path.join(args.out, 'best_corner_detector_openvino_model')
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(exported, dst)
        shutil.rmtree(exported)
        # The app loads the OpenVINO model by these file names
        for ext in ('.xml', '.bin'):
            for f in os.listdir(dst):
                if f.endswith(ext) and f != 'best_corner_detector' + ext:
                    os.replace(os.path.join(dst, f), os.path.join(dst, 'best_corner_detector' + ext))
        run['openvino'] = dst

    with open(os.path.join(args.out, 'run.json'), 'w') as f:
        json.dump(run, f, indent=2)
    print(f'Done -> {args.out}')


if __name__ == '__main__':
    main()
