#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess

def main():
    parser = argparse.ArgumentParser(description="CyAnno OmniBenchmark module")

    # Required OmniBenchmark arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where outputs must be written.")
    parser.add_argument("--name", type=str, required=True,
                        help="Dataset name used in output filename.")

    # Inputs defined in YAML
    parser.add_argument("--data.matrix", dest="matrix", type=str, required=True)
    parser.add_argument("--data.true_labels", dest="labels", type=str, required=True)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.name}_predicted_labels.txt"

    print(f"📄 Output will be saved to: {output_file}")

    run_script = Path(__file__).resolve().parents[1] / "cyanno_pipeline" / "run_cyanno.py"

    cmd = [
        "python",
        "-m", "cyanno_pipeline.run_cyanno",
        args.matrix,
        args.labels,
        str(output_file)
    ]

    print("🚀 Running CyAnno pipeline:")
    print("   ", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    print("========== STDOUT ==========")
    print(result.stdout)
    print("========== STDERR ==========")
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"❌ CyAnno crashed (exit {result.returncode})")

    print(f"🎉 SUCCESS — prediction saved to {output_file}")

if __name__ == "__main__":
    main()
