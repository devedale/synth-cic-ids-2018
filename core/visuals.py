#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Analytics Module for AI/ML performance evaluation and data distribution.
Implements a strict, clean academic aesthetic (matplotlib, seaborn).
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np

# Apply strict Academic styling globally
plt.style.use('seaborn-v0_8-paper')
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    'font.family': 'serif',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'lines.linewidth': 1.5,
    'figure.autolayout': True,
    'figure.dpi': 300
})

def plot_dataset_statistics(csv_path: Path, output_dir: Path) -> None:
    """Read crosstab statistics and generate an academic heatmap distribution."""
    if not csv_path.exists():
        print(f"[visuals] Cannot plot. File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path, index_col=0)
    
    # Drop margins for the core visualization if they exist
    plot_df = df.copy()
    
    if 'Total' in plot_df.index:
        plot_df.drop(index=['Total'], inplace=True)
        
    if 'Total' in plot_df.columns:
        plot_df = plot_df.sort_values(by='Total', ascending=False)
        # We KEEP the Total column visible as per user request
        
    # Massive figsize to allow the text to breathe
    fig, ax = plt.subplots(figsize=(24, 14))
                
    ax.set_title("Network Traffic Class Distribution Over Time", pad=20, fontsize=14, weight='bold')
    ax.set_ylabel("Threat Class", weight='bold')
    ax.set_xlabel("Capture Day", weight='bold')
    plt.xticks(rotation=45, ha='right')
    
    out_file = output_dir / "dataset_distribution_heatmap.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_file), bbox_inches='tight')
    plt.close(fig)
    print(f"[visuals] Academic heatmap saved to {out_file}")

def plot_pca_variance(explained_variances: list, output_dir: Path) -> None:
    """Generate Scree plot for PCA Explained Variance Ratio."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    cum_var = np.cumsum(explained_variances)
    x_ticks = range(1, len(explained_variances) + 1)
    
    ax.bar(x_ticks, explained_variances, alpha=0.6, color='#2c3e50', label='Individual Variance')
    ax.step(x_ticks, cum_var, where='mid', color='#e74c3c', linewidth=2, label='Cumulative Variance')
    
    ax.set_title("PCA Scree Plot: Explained Variance", pad=15, weight='bold')
    ax.set_ylabel("Variance Ratio", weight='bold')
    ax.set_xlabel("Principal Component", weight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x_ticks)
    ax.legend(loc='best')
    
    out_file = output_dir / "pca_variance_scree.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_file), bbox_inches='tight')
    plt.close(fig)
    print(f"[visuals] PCA Scree plot saved to {out_file}")

def plot_training_curves(history_dict: dict, output_dir: Path) -> None:
    """Plot Neural Network metrics standard (e.g. loss/val_loss)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for key, values in history_dict.items():
        linestyle = '-' if 'val' not in key else '--'
        ax.plot(values, label=key, linestyle=linestyle)
        
    ax.set_title("Model Training Performance", pad=15, weight='bold')
    ax.set_ylabel("Metric Value", weight='bold')
    ax.set_xlabel("Epoch", weight='bold')
    ax.legend(loc='best')
    
    out_file = output_dir / "model_training_curves.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_file), bbox_inches='tight')
    plt.close(fig)
    print(f"[visuals] Training curves saved to {out_file}")
