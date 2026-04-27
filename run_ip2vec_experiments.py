#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated testing suite for IP2Vec Sentence permutations.
Iterates over all combinations, modifies settings.py, re-runs the data pipeline,
and runs the centralized ML experiments, accumulating results into a single CSV.
"""
import itertools
import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

# Pin hash seed before any hashing occurs — also inherited by subprocesses
os.environ["PYTHONHASHSEED"] = "42"

from configs.settings import RANDOM_SEED, set_global_seed

SETTINGS_FILE = Path("configs/settings.py")
RESULTS_CSV = Path("results/centralized_results.csv")
FULL_REPORT_CSV = Path("results/full_comparison_report.csv")

# 1. Define the combinations to test (excluding Src/Dst Region alone as requested)
COMBINATIONS = [
    ["Dst Port", "Protocol", "Src Region"],
    ["Dst Port", "Protocol"],
    ["Src IP", "Dst IP", "Dst Port", "Protocol"]
]

def generate_permutations():
    sentences = []
    for f_list in COMBINATIONS:
        for perm in itertools.permutations(f_list):
            sentences.append(list(perm))
    return sentences

def patch_settings(sentence: list):
    """Replaces the IP2VEC_SENTENCE line in configs/settings.py"""
    with open(SETTINGS_FILE, "r") as f:
        lines = f.readlines()
        
    with open(SETTINGS_FILE, "w") as f:
        for line in lines:
            if line.startswith("IP2VEC_SENTENCE ="):
                f.write(f'IP2VEC_SENTENCE = {sentence}\n')
            else:
                f.write(line)

def run_experiment(sentence: list, is_first: bool):
    print(f"\n{'#'*80}")
    print(f"### RUNNING PERMUTATION: {sentence}")
    print(f"{'#'*80}\n")
    
    # Determinism env inherited by subprocesses
    env = {**os.environ, "PYTHONHASHSEED": str(RANDOM_SEED),
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}

    # 1. Update settings.py
    patch_settings(sentence)
    
    # 2. Run Data Pipeline (main.py) to regenerate ip2vec_embeddings parquet
    print(">> Step 1/2: Regenerating Data Pipeline (main.py)...")
    subprocess.run([sys.executable, "main.py"], check=True, env=env)
    
    # 3. Run Centralized Experiments
    # Note: This will use the existing HPO best_params.json!
    print(">> Step 2/2: Running Centralized Experiments...")
    subprocess.run([sys.executable, "run_experiment_centralized.py"], check=True, env=env)
    
    # 4. Append results to the full comparison report
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        
        # If it's the very first run, we create the new file with headers
        if is_first:
            df.to_csv(FULL_REPORT_CSV, index=False)
        else:
            # Otherwise we append without writing the header
            df.to_csv(FULL_REPORT_CSV, mode='a', header=False, index=False)
        print(f"✅ Appended {len(df)} rows to {FULL_REPORT_CSV}")
    else:
        print(f"❌ Error: {RESULTS_CSV} not found after run.")

if __name__ == "__main__":
    sentences = generate_permutations()
    print(f"Total permutations to test: {len(sentences)}")
    
    # Save a backup of settings.py
    backup = SETTINGS_FILE.read_text()
    
    Path("results").mkdir(exist_ok=True)
    if FULL_REPORT_CSV.exists():
        FULL_REPORT_CSV.unlink() # Delete old report if exists
        
    try:
        for i, sentence in enumerate(sentences):
            run_experiment(sentence, is_first=(i == 0))
            
    finally:
        # Restore settings.py
        print("\nRestoring original configs/settings.py...")
        SETTINGS_FILE.write_text(backup)
        print(f"\n🎉 All tests completed! Full report saved to: {FULL_REPORT_CSV}")
