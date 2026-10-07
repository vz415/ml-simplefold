"""Populate inference caches without constructing models or doing sampling."""
import argparse
import os
from pathlib import Path
import subprocess


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        print(f"Cached: {destination}", flush=True)
        return
    partial = destination.with_name(destination.name + ".part")
    print(f"Downloading: {url}", flush=True)
    subprocess.run([
        "curl", "--fail", "--location", "--retry", "5", "--retry-delay", "5",
        "--connect-timeout", "30", "--continue-at", "-", "--output", str(partial), url,
    ], check=True)
    if not partial.stat().st_size:
        raise RuntimeError(f"Empty download: {url}")
    partial.replace(destination)


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Download the large model weights inside a Slurm job.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--simplefold_model", default="simplefold_100M", choices=[
        "simplefold_100M", "simplefold_360M", "simplefold_700M",
        "simplefold_1.1B", "simplefold_1.6B", "simplefold_3B",
    ])
    args = parser.parse_args()
    artifacts = args.artifact_dir.resolve()
    download(f"https://ml-site.cdn-apple.com/models/simplefold/{args.simplefold_model}.ckpt",
             artifacts / f"checkpoints/{args.simplefold_model}.ckpt")
    model = "esm2_t36_3B_UR50D"
    download(f"https://dl.fbaipublicfiles.com/fair-esm/models/{model}.pt",
             artifacts / f"torch/hub/checkpoints/{model}.pt")
    download(f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model}-contact-regression.pt",
             artifacts / f"torch/hub/checkpoints/{model}-contact-regression.pt")
    download("https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl",
             artifacts / "ccd/ccd.pkl")


if __name__ == "__main__":
    main()
