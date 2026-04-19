#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project-wide utility functions for File I/O and Statistics to prevent core modules bloat."""

from pathlib import Path

def clean_temp(base_dir: Path) -> None:
    """Cleanup temporary extraction and flow-csv folders used by ingestion."""
    import shutil
    for rel in ("spark_tmp",):
        target = base_dir / rel
        if target.exists():
            for path in target.iterdir():
                if path.is_file():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)

def generate_crosstab_report(full_df, stats_file: Path) -> None:
    """Generate a Cross-Tabulated Dataset Statistics Report explicitly utilizing PySpark actions."""
    if stats_file.exists() or full_df.count() == 0:
        return
        
    print("\n[utils] Generating Cross-Tabulated Dataset Statistics Report (First run)...")
    
    # Generate the crosstab using PySpark (fast distributed action)
    crosstab_df = full_df.crosstab("Label", "_source_day")
    
    # Convert to Pandas for computing margins and clean printing (matrix is extremely small, ~15x10)
    crosstab_pd = crosstab_df.toPandas()
    
    if not crosstab_pd.empty:
        crosstab_pd.set_index("Label__source_day", inplace=True)
        crosstab_pd.index.name = "Label"
        
        # Calculate right margin (Row Totals)
        crosstab_pd["Total"] = crosstab_pd.sum(axis=1)
        
        # Calculate bottom margin (Column Totals)
        crosstab_pd.loc["Total"] = crosstab_pd.sum(axis=0)
        
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        crosstab_pd.to_csv(stats_file)
        
        print("\n[utils] --- Dataset Distribution Statistics ---")
        print(crosstab_pd.to_string())
        print(f"----------------------------------------------\n[utils] Saved to {stats_file}\n")
