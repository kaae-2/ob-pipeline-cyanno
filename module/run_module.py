import subprocess
from pathlib import Path

def run(input_files, output_files, params, **kwargs):
    """
    Python wrapper used by the OmniBenchmark orchestrator (Snakefile).
    This must match the input/output keys defined in your benchmark plan.
    """

    # 1. Update keys to match your entrypoint and benchmark YAML
    train_matrix = input_files["data.train_matrix"]
    train_labels = input_files["data.train_labels"]
    test_matrix = input_files["data.test_matrix"]

    # 2. Get the output path (ensure the key matches your Snakefile/config)
    # Using 'analysis.prediction.cyannotool' as per your previous setup
    pred_path = Path(output_files["analysis.prediction.cyannotool"])
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Path to the actual runner script
    # parents[1] goes from 'module/' up to the project root
    run_script = Path(__file__).resolve().parents[1] / "cyanno_pipeline" / "run_cyanno.py"

    # 4. Construct the command to match run_cyanno.py expectations:
    # Usage: run_cyanno.py <train_matrix> <train_labels> <test_matrix_tar> <output_tar>
    cmd = [
        "python",
        str(run_script),
        str(train_matrix),
        str(train_labels),
        str(test_matrix),
        str(pred_path)
    ]

    print("OmniBenchmark Orchestrator running CyAnno...")
    print("Running:", " ".join(cmd))
    
    subprocess.check_call(cmd)