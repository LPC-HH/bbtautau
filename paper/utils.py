from __future__ import annotations

import pandas as pd
from boostedhh.hh_vars import years as all_years

label_maps = {
    "hh": r"\tauhh",
    "hm": r"\tauhm",
    "he": r"\tauhe",
    "bbtt": r"ggF \HHbbtt",
    "vbfbbtt": r"VBF \HHbbtt",
    "vbfbbtt-k2v0": r"VBF \HHbbtt $\kapvv=0$",
}

all_signals = ["bbtt", "vbfbbtt", "vbfbbtt-k2v0"]


def csv_to_latex_successive_removal_notable(
    csv_paths: dict[str, dict[str, str]],
    channel: str,
    signals: list[str] = all_signals,
    years: list[str] = all_years,
) -> str:
    """Converts successive removal CSVs into a LaTeX table, BUT without the table environment.

    Args:
        csv_path: Path to the CSV file
        output_path: Path to save the LaTeX output (optional)
        caption: Table caption (optional)
        label: Table label for referencing (optional)

    Returns:
        LaTeX table code as a string
    """

    # [ f"/home/users/lumori/bbtautau/plots/TriggerStudy/25Apr16/{year}/{signal}/progressive_removal_{channel}_.csv" for year in all_years for signal in ["bbtt", "vbfbbtt", "vbfbbtt-k2v0"]]

    dfs = {
        signal: {year: pd.read_csv(csv_paths[signal][year]) for year in years} for signal in signals
    }

    latex_table = ""

    for signal in dfs:
        for i, (year, df) in enumerate(dfs[signal].items()):
            latex_table += r"\resizebox*{1\textwidth}{!}{"
            latex_table += "\\begin{tabular}{" + "c" * len(df.columns) + "}\n"
            latex_table += "\\hline\n"

            if i == 0:
                latex_table += "\\hline\n"
                latex_table += (
                    r"\multicolumn{"
                    + str(len(df.columns))
                    + r"}{c}{\textbf{"
                    + f"{label_maps[signal]}, {label_maps[channel]}"
                    + r"}} \\"
                )
                latex_table += "\\hline\n"

            # Add headers
            latex_table += (
                r"\textbf{"
                + f"{year}"
                + r"} & "
                + r" & ".join(
                    str(val)
                    .replace("_", "\_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("\_{", "_{")
                    .replace("Trigger Efficiency", "")
                    for val in df.columns[1:]
                )
                + " \\\\\n"
            )
            latex_table += "\\hline\n"

            # Add data rows
            for _, row in df.iterrows():
                latex_table += (
                    " & ".join(
                        str(val).replace("%", "\%")
                        #   .replace(", XbbvsQCD > 0.95", "QCD cut")
                        #   .replace(", XbbvsQCDTop > 0.95", "QCDTop cut")
                        #   .replace("(> 250)","")
                        #   .replace("(>250, >200)","")
                        .replace(">", "$>$")
                        .replace("boosted", "")
                        .replace(" (", r", \pt (")
                        .replace(")", r")\GeV")
                        for val in row
                    )
                    + " \\\\\n"
                )

            latex_table += "\\hline\n"
            latex_table += "\\end{tabular}\n"
            latex_table += "}\n"
            if i != len(dfs[signal]) - 1:
                latex_table += "\\vspace{2mm}\n"

    return latex_table
