import csv
import sys
import re
import glob
import os

def format_float(val):
    try:
        return f"{float(val):.4f}"
    except ValueError:
        return val  # return as is if it's not a float

def extract_caption_from_filename(filename):
    match = re.search(r"results_(.+)_(\d+(?:\.\d+)?)\.csv", filename)
    if match:
        map_name = match.group(1)
        speed = match.group(2)
        return f"Performance metrics on ${map_name}$ at speed {speed}."
    else:
        match = re.search(r"resultsangle_(.+)_(\d+(?:\.\d+)?)\.csv", filename)
        if match:
            map_name = match.group(1)
            speed = match.group(2)
            return f"Performance metrics for angle estimation on ${map_name}$ at speed {speed}."
        match = re.search(r"results_(.+)_(\d+(?:\.\d+)?)\.csv", filename)
        if match:
            map_name = match.group(1)
            speed = match.group(2)
            return f"Performance metrics on ${map_name}$ at speed {speed}."
        
        
    return "Caption goes here."

def csv_to_latex_table(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)

        # Custom column name mapping (optional)
        column_rename = {
            "Method": r"\textbf{Method}",
            "Laps Completed": r"\textbf{Laps Completed}",
            "Mean": r"\textbf{Mean Error}",
            "Median": r"\textbf{Median}",
            "Max": r"\textbf{Max}",
            "Min": r"\textbf{Min}",
            "Std Dev": r"\textbf{Std Dev}",
        }

        latex = []
        latex.append(r"\begin{table}[!ht]")
        latex.append(r"    \centering")
        latex.append(r"    \begin{tabularx}{\textwidth}{|l|X|c|c|c|c|c|}")
        latex.append(r"    \hline")

        # Header row
        header_row = " & ".join([column_rename.get(h, h) for h in headers]) + r" \\ \hline"
        latex.append("    " + header_row)

        for row in reader:
            formatted_row = [format_float(cell).replace('"', '') for cell in row]
            line = " & ".join(formatted_row) + r" \\ \hline"
            latex.append("    " + line)

        latex.append(r"    \end{tabularx}")
        latex.append(f"    \\caption{{{extract_caption_from_filename(os.path.basename(filename))}}}")
        latex.append(r"\end{table}")

        return "\n".join(latex)

def process_directory(directory, output_file):
    csv_files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    all_latex = []
    for csv_file in csv_files:
        all_latex.append(csv_to_latex_table(csv_file))
        all_latex.append("\n")  # Separate tables by a newline

    with open(output_file, "w") as f:
        f.write("\n".join(all_latex))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_to_latex.py <directory_with_csvs> <output_file.txt>")
        sys.exit(1)

    process_directory(sys.argv[1], sys.argv[2])
