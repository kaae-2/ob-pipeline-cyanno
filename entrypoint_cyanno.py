#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="CyAnno OmniBenchmark module")

    # Required OmniBenchmark arguments
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory where outputs must be written."
    )
    parser.add_argument(
        "--name", type=str, required=True,
        help="Dataset name used in output filename."
    )

    # Inputs defined in YAML (Matching your requested flags)
    parser.add_argument("--data.train_matrix", dest="train_matrix", type=str, required=True)
    parser.add_argument("--data.train_labels", dest="train_labels", type=str, required=True)
    parser.add_argument("--data.test_matrix", dest="test_matrix", type=str, required=True)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # OUTPUT: Must match benchmark YAML: {dataset}_predicted_labels.tar.gz
    output_file = output_dir / f"{args.name}_predicted_labels.tar.gz"
    print(f"Output will be saved to: {output_file}", flush=True)

    # Repo root in the cloned module
    repo_root = Path(__file__).resolve().parent
    run_script = repo_root / "cyanno_pipeline" / "run_cyanno.py"

    cmd = [
        sys.executable,
        str(run_script),
        args.train_matrix,
        args.train_labels,
        args.test_matrix,
        str(output_file),
    ]

    print("Running CyAnno pipeline:")
    print("   ", " ".join(cmd), flush=True)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"CyAnno crashed (exit {result.returncode})")

    print(f"SUCCESS — prediction saved to {output_file}", flush=True)

if __name__ == "__main__":
    main()