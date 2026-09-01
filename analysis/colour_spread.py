"""Analysis script: summarize reflected-light readings by colour for each run.

For every CSV file found under `results_dir` (grouped by the run directory
name), computes descriptive statistics (count, mean, std, min, max, median) of
the "reflected_avg" column per "colour" value, prints the summary, and writes
it to a "<run>_reflected_avg_by_colour.csv" file.
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd

results_dir = Path("results/colour-run")

# Collect dataframes for each run
run_data = defaultdict(list)

for csv_file in results_dir.rglob("*.csv"):
    run = csv_file.parts[1]  # e.g. colour_recognition_run
    df = pd.read_csv(csv_file)

    # Only keep the columns we care about
    run_data[run].append(df[["colour", "reflected_avg"]])

for run, dfs in run_data.items():
    combined = pd.concat(dfs, ignore_index=True)

    summary = (
        combined
        .groupby("colour")["reflected_avg"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            min="min",
            max="max",
            median="median",
        )
        .reset_index()
    )

    print(f"\n=== {run} ===")
    print(summary)

    summary.to_csv(f"{run}_reflected_avg_by_colour.csv", index=False)