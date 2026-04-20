import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from core.visuals import plot_metrics_radar

def generate_report():
    out_dir = Path("results")
    vis_dir = out_dir / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    cent_path = out_dir / "centralized_results.csv"
    fed_path = out_dir / "federated_results.csv"
    
    if not cent_path.exists() or not fed_path.exists():
        print("Missing one of the results CSV files (centralized or federated). Please run experiments first.")
        return
        
    df_cent = pd.read_csv(cent_path)
    df_cent["paradigm"] = "centralized"
    df_fed = pd.read_csv(fed_path)
    df_fed["paradigm"] = "federated"
    
    df_cent["display_name"] = df_cent["config_name"] + " (C)"
    df_fed["display_name"] = df_fed["config_name"] + " (F)"
    
    df_all = pd.concat([df_cent, df_fed], ignore_index=True)
    df_all.to_csv(out_dir / "full_comparison_report.csv", index=False)
    print(f"Saved full comparison report to {out_dir / 'full_comparison_report.csv'}")
    
    metrics_to_plot = ["accuracy", "f1", "fpr", "fnr", "auc_roc"]
    sns.set_theme(style="whitegrid", context="paper")
    
    for m in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_all, x="config_name", y=m, hue="paradigm", ax=ax, palette="deep")
        ax.set_title(f"Comparison of {m.upper()} across configurations", pad=20, weight="bold")
        ax.set_ylabel(m)
        ax.set_xlabel("Configuration")
        
        plt.legend(title="Paradigm", loc="upper right", bbox_to_anchor=(1.15, 1))
        fig.savefig(str(vis_dir / f"metrics_barchart_{m}.pdf"), bbox_inches="tight")
        plt.close(fig)
        
    print(f"Saved barcharts to {vis_dir}")
    
    df_radar = df_all.copy()
    df_radar["config_name"] = df_radar["display_name"]
    plot_metrics_radar(df_radar, vis_dir)

if __name__ == "__main__":
    generate_report()
