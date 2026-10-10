"""Analysis script: summarize reflected-light readings by colour for each run.

For every CSV file found under `results_dir` (grouped by the run directory
name), computes descriptive statistics (count, mean, std, min, max, median) of
each ground sensor's reading ("reflected_0" = left, "reflected_1" = right)
per "colour" value, prints the summaries, and writes them to
"<run>_reflected_<sensor>_by_colour.csv" files.

Colour recognition only uses one sensor (see OptionGroundSensor), so the
option_centers / allowed_offsets in colour_calibration.yaml should be taken
from the summary of the sensor configured for that robot.
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd

results_dir = Path("results/colour-run")

SENSOR_COLUMNS = ["reflected_0", "reflected_1"]

# Collect dataframes for each run
run_data = defaultdict(list)

for csv_file in results_dir.rglob("*.csv"):
    run = csv_file.parts[1]  # e.g. colour_recognition_run
    df = pd.read_csv(csv_file)

    # Only keep the columns we care about
    run_data[run].append(df[["colour", *SENSOR_COLUMNS]])

for run, dfs in run_data.items():
    combined = pd.concat(dfs, ignore_index=True)

    for column in SENSOR_COLUMNS:
        summary = (
            combined
            .groupby("colour")[column]
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

        print(f"\n=== {run}: {column} ===")
        print(summary)

        summary.to_csv(f"{run}_{column}_by_colour.csv", index=False)