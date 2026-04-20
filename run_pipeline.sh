#!/bin/bash
set -e

echo "========================================================="
echo "  CICIDS-2018 Experimental Pipeline Orchestrator"
echo "========================================================="

echo "[1/6] Running Setup..."
# Install the package in editable mode and ensure requirements are met
./.venv/bin/pip install -e .
./.venv/bin/pip install -r requirements.txt

echo -e "\n[2/6] Running Data Ingestion & Preprocessing (main.py)..."
./.venv/bin/python main.py

echo -e "\n[3/6] Running Hyperparameter Optimization (run_hpo.py)..."
./.venv/bin/python run_hpo.py

echo -e "\n[4/6] Running Centralized Experiments (run_experiment_centralized.py)..."
./.venv/bin/python run_experiment_centralized.py

echo -e "\n[5/6] Running Federated Learning Experiments (run_experiment_federated.py)..."
./.venv/bin/python run_experiment_federated.py

echo -e "\n[6/6] Generating Final Reports & Visualizations (run_report.py)..."
./.venv/bin/python run_report.py

echo -e "\n========================================================="
echo "  Pipeline Completed Successfully!"
echo "  Check results/ and results/visuals/ for the outputs."
echo "========================================================="
