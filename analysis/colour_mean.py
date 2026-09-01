"""Compute per-file and per-run mean/std statistics over result CSVs.

Recursively reads every CSV file under ``results/``, computes the mean
and standard deviation of each numeric column per file, and groups the
per-file statistics into overall statistics per "run" (the top-level
subdirectory of each CSV file under ``results/``, e.g. ``black-run``,
``white-run``). Prints both summaries and writes them to
``results_summary.csv`` (per-file) and ``run_summary.csv`` (per-run) in
the current working directory.
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd

results_dir = Path("results")

# Per-file summary
summary = []

# Store dataframes for each run
run_data = defaultdict(list)

for csv_file in results_dir.rglob("*.csv"):
    df = pd.read_csv(csv_file)

    run = csv_file.parts[1]  # black-run, white-run, etc.

    # Save for overall run statistics
    run_data[run].append(df)

    # Per-file statistics
    row = {
        "run": run,
        "file": csv_file.name,
    }

    for col in df.columns:
        row[f"{col}_mean"] = df[col].mean()
        row[f"{col}_std"] = df[col].std()

    summary.append(row)

# DataFrame of per-file statistics
summary_df = pd.DataFrame(summary)

# Overall statistics for each run
run_summary = []

for run, dfs in run_data.items():
    combined = pd.concat(dfs, ignore_index=True)

    row = {"run": run}
    for col in combined.columns:
        row[f"{col}_mean"] = combined[col].mean()
        row[f"{col}_std"] = combined[col].std()

    run_summary.append(row)

run_summary_df = pd.DataFrame(run_summary)

print("Per-file statistics:")
print(summary_df)

print("\nOverall statistics by run:")
print(run_summary_df)

summary_df.to_csv("results_summary.csv", index=False)
run_summary_df.to_csv("run_summary.csv", index=False)