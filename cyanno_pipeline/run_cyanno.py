#!/usr/bin/env python3
import sys
import os
import gzip
import tarfile
import tempfile
import pandas as pd
import warnings
from pathlib import Path
from typing import List

# --- Make sure `cyanno_pipeline` can be imported ---
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cyanno_pipeline.cyanno import CyAnnoClassifier

# --- Helper to load specific file types ---
def load_dataframe(path, header=None, names=None):
    """
    Robust loader that handles:
    1. Standard .csv.gz
    2. .tar.gz (extracts the first CSV found)
    3. .csv (plain text)
    """
    path = str(path)
    if path.endswith(".tar.gz") or path.endswith(".tar"):
        try:
            with tarfile.open(path, "r:*") as tar:
                csv_member = next((m for m in tar.getmembers() if m.name.endswith('.csv')), None)
                if not csv_member:
                    raise ValueError(f"No CSV found in {path}")
                f = tar.extractfile(csv_member)
                return pd.read_csv(f, header=header, names=names)
        except Exception:
            # If tar fails, try treating it as a misnamed gzip
            return pd.read_csv(path, compression='gzip', header=header, names=names)
    else:
        # Assume gzip or plain text
        try:
            return pd.read_csv(path, header=header, names=names)
        except:
            return pd.read_csv(path, compression='gzip', header=header, names=names)

def main(train_matrix_path, train_labels_path, test_matrix_tar_path, output_path):
    # 1. Prepare Data
    print(f"Loading Training Data...")
    
    # Load Training Matrix (No header based on your previous input files)
    train_matrix_df = load_dataframe(train_matrix_path, header=None)
    
    # Load Training Labels (Force column name 'cell_type')
    train_labels_df = load_dataframe(train_labels_path, header=None, names=["cell_type"])

    # Sanity Check
    if len(train_matrix_df) != len(train_labels_df):
        raise ValueError(f"Mismatch: {len(train_matrix_df)} train cells vs {len(train_labels_df)} labels")

    # Merge for training
    train_df = pd.concat([train_matrix_df, train_labels_df], axis=1).dropna(subset=["cell_type"])
    
    # Generate generic marker names (M0, M1...) since files have no header
    marker_names = [f"M{i}" for i in range(train_matrix_df.shape[1])]
    train_df.columns = marker_names + ["cell_type"]

    # 2. Train Model
    print("Training CyAnno Model...")
    clf = CyAnnoClassifier(markers=marker_names)
    clf.train(train_df)

    # 3. Process Test Data (Iterating over TAR contents)
    print(f"Processing Test Matrices from: {test_matrix_tar_path}")
    
    # Remove existing output if it exists (safety)
    if os.path.exists(output_path):
        os.remove(output_path)

    # Use a temporary directory to store individual prediction files before tarring
    with tempfile.TemporaryDirectory() as tmpdir:
        output_files: List[str] = []

        # Open the input TAR containing test matrices
        with tarfile.open(test_matrix_tar_path, "r:*") as tar_in:
            # Filter for CSV files inside the tar
            test_members = [m for m in tar_in.getmembers() if m.name.endswith('.csv') and m.isfile()]

            if not test_members:
                raise ValueError("No CSV files found in test matrix archive!")

            for member in test_members:
                print(f"   Predicting: {member.name}")
                
                # Extract and load the specific test file
                f = tar_in.extractfile(member)
                test_sample_df = pd.read_csv(f, header=None)
                
                # Ensure columns match training
                if test_sample_df.shape[1] != len(marker_names):
                    warnings.warn(f"   Warning: Column count mismatch in {member.name}. Expected {len(marker_names)}")
                test_sample_df.columns = [f"M{i}" for i in range(test_sample_df.shape[1])]

                # Predict
                preds, _ = clf.predict(test_sample_df)

                # Format predictions
                # Note: dgcytof uses float formatting, but CyAnno might output strings/integers.
                # We simply convert to string to be safe.
                output_labels = [str(p) for p in preds]

                # Create output filename: e.g., sample_1.csv -> sample_1.predictions.csv.gz
                safe_name = os.path.basename(member.name)
                if safe_name.endswith(".csv"):
                    safe_name = safe_name.replace(".csv", ".predictions.csv.gz")
                else:
                    safe_name = f"{safe_name}.predictions.csv.gz"

                file_path = os.path.join(tmpdir, safe_name)

                # Save individual result as GZIP (No header, no index)
                with gzip.open(file_path, "wt") as handle:
                    pd.Series(output_labels).to_csv(handle, index=False, header=False)
                
                output_files.append(file_path)

        # 4. Create Final Output TAR
        print(f"Packing results to {output_path}...")
        with tarfile.open(output_path, "w:gz") as tar_out:
            for path in output_files:
                tar_out.add(path, arcname=os.path.basename(path))

    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: run_cyanno.py <train_matrix> <train_labels> <test_matrix_tar> <output_tar>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])