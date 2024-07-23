from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import subprocess
import argparse

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
    return parser.parse_args()


def extract_args_from_function(args):
    """
    Only keep the name of the arguments
    """
    return [arg.split(":")[0].strip() for arg in args.split(",")]


def extract_return_from_function(return_type):
    if return_type == "":
        return None
    return return_type.strip()


def extract_function_from_code(code):
    reg_exp = re.compile(r"def\s+(\w+)\s*\(([\s\S]*?)\)\s*(?:->\s*([^\{\n]+))?\s*:")
    matches = reg_exp.findall(code)
    functions = []
    for match in matches:
        function = {}
        function["name"] = match[0]
        function["args"] = extract_args_from_function(match[1])
        function["return"] = extract_return_from_function(match[2])
        function["typed_args"] = list(map(is_arg_typed, match[1].split(",")))
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


def get_percent_typed_args(*python_file_paths, progress_bar=False):
    data_frames = []

    with Progress() as progress:
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
            dataframe["number of args"] = dataframe["args"].apply(lambda x: len(x))
            data_frames.append(dataframe)

    return pd.concat(data_frames)


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
    output = f"Total percent of typed arguments in all python files: {total_percent:.2%}"
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
    print(output)


if __name__ == "__main__":
    main()
