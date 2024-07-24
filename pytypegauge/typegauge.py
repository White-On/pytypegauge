from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import contextlib
import subprocess
import argparse
from .logger import logger
from io import BytesIO

from rich.progress import Progress


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the typing of python code in a specified directory"
    )
    parser.add_argument(
        "directory", type=str, help="the specified directory", default="."
    )
    parser.add_argument(
        "-g",
        "--git",
        action="store_true",
        help="Analyze only files tracked by git",
    )
    parser.add_argument(
        "-p",
        "--plot-output",
        action="store_true",
        help="Display a graph of distribution of the problem found in the code",
    )
    parser.add_argument(
        "-csv",
        "--csv-output",
        type=str,
        help="Name of the csv file where the results will be saved",
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
        help="Only return the percentage of typed arguments (useful for scripts)",
    )
    parser.add_argument(
        "-f",
        "--full-report",
        action="store_true",
        help="Create a full report with all the functions that are not typed",
    )
    return parser.parse_args()


def extract_args_from_function(args: str) -> list:
    """
    Only keep the name of the arguments
    """
    splited_args = [arg.split(":")[0].strip() for arg in args.split(",")]
    arg_empty = splited_args == [""]
    if arg_empty:
        return []
    return splited_args


def extract_return_from_function(return_type: str) -> str:
    if return_type == "":
        return "<no-return>"
    return return_type.strip()


def extract_function_from_code(code: str) -> list:
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


def is_arg_typed(arg: str) -> bool:
    """
    the function checks if the argument is followed by a type
    so if there is a : or a = in the argument then it is typed
    """
    if "cls" in arg or "self" in arg:
        return True
    return ":" in arg or "=" in arg


def get_git_files(directory: Path) -> list:
    result_command = subprocess.run(
        ["git", "ls-files"], cwd=directory, capture_output=True, text=True
    )
    tracked_files = result_command.stdout.splitlines()

    return [directory / file for file in tracked_files]

def typage_distribution(function_dataframe: pd.DataFrame) -> dict:
    plot_df = function_dataframe.copy()
    # add a collumn to the dataframe named problem with a value depending on a condition
    # 1 if some of the arguments are not typed 
    # 2 if the return is not typed
    # 3 if both are not typed
    some_args_not_typed = plot_df["typed_args"].apply(lambda args: any(not arg for arg in args))
    return_not_typed = plot_df["return"] == "<no-return>"
    both_not_typed = some_args_not_typed & return_not_typed
    plot_df.loc[:, "problem"] = 0
    plot_df.loc[both_not_typed, "problem"] = 3
    plot_df.loc[return_not_typed & ~both_not_typed, "problem"] = 2
    plot_df.loc[some_args_not_typed & ~both_not_typed, "problem"] = 1
    plot_df = plot_df.drop(columns=["typed_args","return","args","number of typed args", "number of args"])
    # we try to get the distribution of all problem accross all files
    # so for each file we get the number of problem 0 1 2 3
    #  att the end we expect a dictionary with the name of the file as key and a list containing the number of occurence of each problem
    # 0 1 2 3
    # we will then plot the distribution of the problem
    problem_distribution = {file_name: [0] * 4 for file_name in plot_df["file"].unique()}
    group_file_problem_distribution = plot_df.groupby(['file', 'problem']).size().unstack(fill_value=0).T.to_dict()
    for file_name, problems in group_file_problem_distribution.items():
        problem_distribution[file_name] = list(problems.values()) 
    
    return problem_distribution

def survey(results: dict, category_names: list) -> tuple:
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    unormalized_data = np.array(list(results.values()), dtype=object)
    unormalized_data[unormalized_data == 0] = ''
    data = np.array(list({key: np.round(distib/np.sum(distib), 3) for key, distib in results.items()}.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, data.shape[1])[::-1])

    fig, ax = plt.subplots(figsize=(10, 5 + len(results) * 0.4))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color, labels=unormalized_data[:, i])
    ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax

def generate_full_report(function_dataframe: pd.DataFrame) -> None:
    condition = function_dataframe["typed_args"].apply(lambda args: any(not arg for arg in args)) | (function_dataframe["return"] == "<no-return>")

    not_typed_df = function_dataframe[condition]
    is_typed_dataframe_empty = not_typed_df.empty
    if not is_typed_dataframe_empty:
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

        # we get the distribution of the problem
        problem_distribution = typage_distribution(function_dataframe)
        survey(problem_distribution, list(problem_code_and_explanation.values()))
        svg_buffer = BytesIO()
        plt.savefig(svg_buffer, format='svg')
        plt.close()
        svg_data = svg_buffer.getvalue().decode('utf-8')

    else:
        markdown_table = "No problem found in the code"
        logger.debug("No problem found in the code")
        svg_data = "No problem found in the code"
    # create full_report.md
    with open("full_report.md", "w") as f:
        f.write("# Full report\n")
        f.write(f"- Total number of functions: **{function_dataframe.shape[0]}**\n")
        f.write(f"- Total number of typed args: **{function_dataframe['number of typed args'].sum()}**\n")
        f.write(f"- Total number of args: **{function_dataframe['number of args'].sum()}**\n")
        f.write(f"- Total percent of typed arguments: **{function_dataframe['number of typed args'].sum() / function_dataframe['number of args'].sum():.2%}**\n")
        f.write("\n")
        f.write("### Explanation Table\n")
        f.write("\n")
        f.write(markdown_table)
        f.write("\n")
        f.write("### Distribution of the problem\n")
        f.write("\n")
        f.write("The distribution of the problem is the number of functions that have a specific problem\n")
        f.write("\n")
        f.write(svg_data)

def get_percent_typed_args(*python_file_paths:Path, progress_bar=False) -> pd.DataFrame:
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
                continue
            dataframe["number of typed args"] = dataframe["typed_args"].apply(
                lambda x: sum(x)
            )
            dataframe["file"] = python_file_path.name
            dataframe["number of args"] = dataframe["args"].apply(lambda x: len(x))
            data_frames.append(dataframe)

    return pd.concat(data_frames)

def get_color_for_percent(total_percent: float) -> str:
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


def main() -> None:
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
        problem_distribution = typage_distribution(df)
        survey(problem_distribution, ["No problem", "Some arguments are not typed", "The return is not typed", "Some arguments and the return are not typed"])
        plt.show()
    if args.full_report:
        generate_full_report(df)
    logger.info(output, extra={"markup": True, "highlighter": None})


if __name__ == "__main__":
    main()
