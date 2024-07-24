from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import contextlib
import subprocess
import argparse
from .logger import logger

from rich.progress import Progress


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse le typage du code python dans un répertoire spécifié"
    )
    parser.add_argument(
        "directory", type=str, help="Le répertoire à analyser", default="."
    )
    parser.add_argument(
        "-g",
        "--git",
        action="store_true",
        help="Analyse uniquement les fichiers suivis par git",
    )
    parser.add_argument(
        "-p",
        "--plot-output",
        action="store_true",
        help="Affiche un graphique des résultats",
    )
    parser.add_argument(
        "-csv",
        "--csv-output",
        type=str,
        help="Nom du fichier CSV pour sauvegarder les résultats",
        default=None,
    )
    parser.add_argument(
        "-md",
        "--markdown-output",
        action="store_true",
        help="Output the results in a special progress markdown format (made for readme.md on github and hooks on github actions)",
    )
    parser.add_argument(
        "-c",
        "--clean-output",
        action="store_true",
        help="Renvoie uniquement le pourcentage de typage des arguments",
    )
    parser.add_argument(
        "-f",
        "--full-report",
        action="store_true",
        help="Renvoie un rapport complet",
    )
    return parser.parse_args()


def extract_args_from_function(args):
    """
    Only keep the name of the arguments
    """
    splited_args = [arg.split(":")[0].strip() for arg in args.split(",")]
    arg_empty = splited_args == [""]
    if arg_empty:
        return []
    return splited_args


def extract_return_from_function(return_type):
    if return_type == "":
        return "<no-return>"
    return return_type.strip()


def extract_function_from_code(code):
    reg_exp = re.compile(r"def\s+(\w+)\s*\(([\s\S]*?)\)\s*(?:->\s*([^\{\n]+))?\s*:")
    matches = reg_exp.findall(code)
    functions = []
    for match in matches:
        function = {}
        function["name"] = match[0]
        function["args"] = extract_args_from_function(match[1]) if match[1] != "" else []
        function["return"] = extract_return_from_function(match[2]) if match[2] != "" else "<no-return>"
        function["typed_args"] = list(map(is_arg_typed, match[1].split(","))) if match[1] != "" else []
        functions.append(function)
    return functions


def is_arg_typed(arg):
    """
    the function checks if the argument is followed by a type
    so if there is a : or a = in the argument then it is typed
    """
    if "cls" in arg or "self" in arg:
        return True
    return ":" in arg or "=" in arg


def get_git_files(directory):
    result_command = subprocess.run(
        ["git", "ls-files"], cwd=directory, capture_output=True, text=True
    )
    tracked_files = result_command.stdout.splitlines()

    return [directory / file for file in tracked_files]

def generate_full_report(df):
    condition = df["typed_args"].apply(lambda args: any(not arg for arg in args)) | (df["return"] == "<no-return>")

    not_typed_df = df[condition]
    # add a collumn to the dataframe named problem with a value depending on a condition
    # 1 if some of the arguments are not typed 
    # 2 if the return is not typed
    # 3 if both are not typed
    some_args_not_typed = not_typed_df["typed_args"].apply(lambda args: any(not arg for arg in args))
    return_not_typed = not_typed_df["return"] == "<no-return>"
    both_not_typed = some_args_not_typed & return_not_typed
    problem_code_and_explanation = {
        0: "No problem (Should not appear)",
        1: "Some arguments are not typed",
        2: "The return is not typed",
        3: "Some arguments and the return are not typed"
    }
    not_typed_df = not_typed_df.copy()
    not_typed_df.loc[:, "problem"] = 0
    not_typed_df.loc[both_not_typed, "problem"] = 3
    not_typed_df.loc[return_not_typed & ~both_not_typed, "problem"] = 2
    not_typed_df.loc[some_args_not_typed & ~both_not_typed, "problem"] = 1   

    # We remove the columns typed_args,return,args and number of typed args
    # because they are not useful for the user
    clean_report = not_typed_df.drop(columns=["typed_args","return","args","number of typed args", "number of args"])
    #  We replace the problem column with the explanation
    clean_report["problem"] = clean_report["problem"].map(problem_code_and_explanation)
    
    # we suround the name of the file and the name of the function with backticks
    clean_report["file"] = clean_report["file"].apply(lambda x: f"`{x}`")
    clean_report["name"] = clean_report["name"].apply(lambda x: f"`{x}`")
    markdown_table = clean_report.to_markdown(index=False)
    # create full_report.md
    with open("full_report.md", "w") as f:
        f.write("# Full report\n")
        f.write(f"- Total number of functions: **{df.shape[0]}**\n")
        f.write(f"- Total number of typed args: **{df['number of typed args'].sum()}**\n")
        f.write(f"- Total number of args: **{df['number of args'].sum()}**\n")
        f.write(f"- Total percent of typed arguments: **{df['number of typed args'].sum() / df['number of args'].sum():.2%}**\n")
        f.write("\n")
        f.write("### Explanation Table\n")
        f.write("\n")
        f.write(markdown_table)

    


def get_percent_typed_args(*python_file_paths, progress_bar=False):
    data_frames = []

    progress_context = Progress() if progress_bar else contextlib.nullcontext()
    with progress_context as progress:
        if progress_bar:
            task = progress.add_task(
                "[green]Analyzing python files...", total=len(python_file_paths)
            )
        for python_file_path in python_file_paths:
            if progress_bar:
                # progress bar update
                progress.update(task, advance=1)
                # progress.tasks[task].description = f"Analyzing {python_file_path.name}"
            python_code = python_file_path.read_text(errors="ignore")
            functions = extract_function_from_code(python_code)
            dataframe = pd.DataFrame(functions)
            if dataframe.empty:
                # some files may not have any function like __init__.py
                # print(f"No function in {python_file_path.name}")
                continue
            dataframe["number of typed args"] = dataframe["typed_args"].apply(
                lambda x: sum(x)
            )
            dataframe["file"] = python_file_path.name
            dataframe["number of args"] = dataframe["args"].apply(lambda x: len(x))
            data_frames.append(dataframe)

    return pd.concat(data_frames)

def get_color_for_percent(total_percent):
    color_ranges = {
        (0, 0.4): "red",
        (0.4, 0.6): "orange",
        (0.6, 0.8): "yellow",
        (0.8, 1): "green"
    }
    
    for range, color in color_ranges.items():
        if range[0] <= total_percent < range[1]:
            return color
    return "green"  # Default color if no range matches


def main():
    args = parse_arguments()
    path_file = Path(args.directory)
    if args.git:
        tracked_path = get_git_files(path_file)
        python_file_paths = [path for path in tracked_path if path.suffix == ".py"]
        if python_file_paths == []:
            raise FileNotFoundError(
                "No python file tracked by git found in the specified directory ( did you forget to initialize a git repository ?)"
            )
    else:
        python_file_paths = list(path_file.rglob("*.py"))
    if python_file_paths == []:
        raise FileNotFoundError("No python file found in the specified directory")

    df = get_percent_typed_args(*python_file_paths, progress_bar=True)
    total_percent = df["number of typed args"].sum() / df["number of args"].sum()
    color = get_color_for_percent(total_percent)
    output = f"Total percent of typed arguments in all python files: [{color} bold]{total_percent:.2%}[/]"
    if args.markdown_output:
        markdown_element = f"![Progress](https://progress-bar.dev/{total_percent*100:.0f}/?title=typed&width=150&scale=100&suffix=%)"
        output = markdown_element
    if args.clean_output:
        output = total_percent
    if args.csv_output:
        df.to_csv(args.csv_output)
    if args.plot_output:
        plt.bar(df["name"], df["number of typed args"] / df["number of args"])
        plt.xticks(rotation=90)
        plt.show()
    if args.full_report:
        generate_full_report(df)
    logger.info(output, extra={"markup": True, "highlighter": None})


if __name__ == "__main__":
    main()
