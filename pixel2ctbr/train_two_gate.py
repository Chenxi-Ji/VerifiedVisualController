#!/usr/bin/env python3
"""Train the two_gate trajectory specialist (pixel_ctbr_two_gate).
Thin preset around train_bptt.py: warm-starts from the one-gate hover model
(v14), chained-BPTT on the two_gate env, ring loss ramped (new objective),
wide-delay DR (v14 robustness recipe), BN adapting (duplicated-gate visuals
are new content). Output: weights/pixel_ctbr_two_gate.pt (best-by-success).
Run: python pixel2ctbr/train_two_gate.py [extra train_bptt args]"""
import subprocess, sys
args = ["python", "-u", "pixel2ctbr/train_bptt.py",
        "--task", "two_gate",
        "--init", "weights/pixel_ctbr_one_gate.pt",
        "--epochs", "20", "--windows", "100", "--lr", "8e-5",
        "--wide-delay",
        "--out", "weights/pixel_ctbr_two_gate.pt"] + sys.argv[1:]
sys.exit(subprocess.call([sys.executable] + args[1:]))
