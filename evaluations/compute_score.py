"""Legacy wrapper for metric evaluation."""

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    cmd = [sys.executable, "evaluations/evaluate_metrics.py", "--real_dir", args.real_dir, "--fake_dir", args.fake_dir]
    if args.device:
        cmd += ["--device", args.device]
    if args.num_workers is not None:
        cmd += ["--num_workers", str(args.num_workers)]
    if args.batch_size is not None:
        cmd += ["--batch_size", str(args.batch_size)]

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

