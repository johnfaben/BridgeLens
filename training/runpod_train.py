"""Drive a training run on a RunPod pod over SSH.

The pod itself is created and deleted separately (runpodctl, the Runpod
console, or the Runpod MCP tools); this script needs its SSH host and port,
from `runpodctl pod get <pod-id>` or the pod's Connect panel. See
docs/development.md, "Training runbook".

Steps (each is idempotent; run them in order):
    package  bundle training code + an exported dataset into a .tar.gz
    push     copy the bundle to the pod's /workspace volume, unpack, install deps
    start    launch training detached (survives disconnects)
    status   show the tail of the training log and whether it finished
    pull     copy the run's outputs back to training_data/runs/<run>/

Usage:
    python -m training.runpod_train package --dataset 2026-11-10
    python -m training.runpod_train push    --dataset 2026-11-10 --host 1.2.3.4 --port 12345
    python -m training.runpod_train start   --dataset 2026-11-10 --run r1 --host ... --port ... [--what classifier]
    python -m training.runpod_train status  --run r1 --host ... --port ...
    python -m training.runpod_train pull    --run r1 --host ... --port ...
"""

import argparse
import os
import shlex
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(ROOT, 'training_data', 'datasets')
PACKAGES_DIR = os.path.join(ROOT, 'training_data', 'packages')
RUNS_DIR = os.path.join(ROOT, 'training_data', 'runs')

# What the pod needs to train: no Flask app, no uploads
CODE_FILES = [
    'train_classifier_cnn.py',
    'best_corner_classifier_cnn.pt',
    'best_corner_detector.pt',
    'training/__init__.py',
    'training/train_classifier.py',
    'training/train_detector.py',
    'training/runpod/pod_setup.sh',
    'training/runpod/run.sh',
]


def package_path(dataset):
    return os.path.join(PACKAGES_DIR, f'{dataset}.tar.gz')


def ssh_base(args):
    cmd = ['ssh', '-p', str(args.port), '-o', 'StrictHostKeyChecking=accept-new',
           '-o', 'ServerAliveInterval=30']
    if args.key:
        cmd += ['-i', args.key]
    return cmd + [f'{args.user}@{args.host}']


def ssh(args, remote_cmd, check=True):
    return subprocess.run(ssh_base(args) + [remote_cmd], check=check)


def scp(args, src, dst):
    cmd = ['scp', '-P', str(args.port), '-o', 'StrictHostKeyChecking=accept-new']
    if args.key:
        cmd += ['-i', args.key]
    subprocess.run(cmd + [src, dst], check=True)


def remote(args, path):
    return f'{args.user}@{args.host}:{path}'


def cmd_package(args):
    ds = os.path.join(DATASETS_DIR, args.dataset)
    if not os.path.exists(os.path.join(ds, 'dataset.json')):
        sys.exit(f'No exported dataset at {ds} (run training.export_dataset first)')
    os.makedirs(PACKAGES_DIR, exist_ok=True)
    out = package_path(args.dataset)
    with tarfile.open(out, 'w:gz') as tar:
        for rel in CODE_FILES:
            tar.add(os.path.join(ROOT, rel), arcname=f'code/{rel}')
        tar.add(ds, arcname=f'datasets/{args.dataset}')
    print(f'{out} ({os.path.getsize(out) / 1e6:.0f} MB)')


def cmd_push(args):
    pkg = package_path(args.dataset)
    if not os.path.exists(pkg):
        sys.exit(f'{pkg} missing (run package first)')
    ssh(args, 'mkdir -p /workspace/incoming /workspace/datasets /workspace/runs /workspace/legacy')
    print('Uploading...', flush=True)
    scp(args, pkg, remote(args, f'/workspace/incoming/{os.path.basename(pkg)}'))
    ds = shlex.quote(args.dataset)
    ssh(args, f'set -e; cd /workspace; rm -rf code datasets/{ds}; '
              f'tar xzf incoming/{ds}.tar.gz; bash code/training/runpod/pod_setup.sh')


def cmd_start(args):
    env = f'CLS_ARGS={shlex.quote(args.cls_args)} DET_ARGS={shlex.quote(args.det_args)}'
    run = shlex.quote(args.run)
    ssh(args, f'mkdir -p /workspace/runs/{run}; '
              f'{env} nohup bash /workspace/code/training/runpod/run.sh '
              f'{shlex.quote(args.dataset)} {run} {args.what} > /workspace/runs/{run}/nohup.out 2>&1 &')
    print(f'Started run {args.run}; check with: status --run {args.run}')


def cmd_status(args):
    run = shlex.quote(args.run)
    ssh(args, f'cd /workspace/runs/{run} && tail -n {args.lines} train.log; '
              f'if [ -f DONE ]; then echo "== DONE"; elif [ -f FAILED ]; then echo "== FAILED"; '
              f'else echo "== still running"; fi; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader',
        check=False)


def cmd_pull(args):
    run = shlex.quote(args.run)
    tar_name = f'{args.run}-results.tar.gz'
    # Weights, OpenVINO export, metrics and logs; skip the big ultralytics batch images
    ssh(args, f'cd /workspace/runs && tar czf /workspace/incoming/{shlex.quote(tar_name)} '
              f'--exclude="*/yolo/*/*.jpg" --exclude="*/yolo/*/weights/last.pt" {run}')
    os.makedirs(RUNS_DIR, exist_ok=True)
    local = os.path.join(RUNS_DIR, tar_name)
    scp(args, remote(args, f'/workspace/incoming/{tar_name}'), local)
    with tarfile.open(local) as tar:
        tar.extractall(RUNS_DIR, filter='data')
    os.remove(local)
    print(f'Results in {os.path.join(RUNS_DIR, args.run)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='step', required=True)

    def add(name, fn, dataset=False, run=False, conn=True):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if dataset:
            p.add_argument('--dataset', required=True)
        if run:
            p.add_argument('--run', required=True)
        if conn:
            p.add_argument('--host', required=True)
            p.add_argument('--port', required=True, type=int)
            p.add_argument('--user', default='root')
            p.add_argument('--key', help='SSH private key (default: ssh picks)')
        return p

    add('package', cmd_package, dataset=True, conn=False)
    add('push', cmd_push, dataset=True)
    p = add('start', cmd_start, dataset=True, run=True)
    p.add_argument('--what', choices=('classifier', 'detector', 'both'), default='both')
    p.add_argument('--cls-args', default='', help='Extra train_classifier args, e.g. "--epochs 60"')
    p.add_argument('--det-args', default='', help='Extra train_detector args')
    p = add('status', cmd_status, run=True)
    p.add_argument('--lines', type=int, default=25)
    add('pull', cmd_pull, run=True)

    args = parser.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
