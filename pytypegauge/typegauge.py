from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import contextlib
import subprocess
import argparse

from io import BytesIO
from rich.progress import Progress
from .logger import logger


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the typing of python code in a specified directory or python file"
    )
    parser.add_argument(
        "input", type=str, help="the specified directory or python file", default="."
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

    parser.add_argument(
        "-ho",
        "--hooks",
        action="store_true",
        help="Output the results in a special progress markdown format (made for hooks on github actions)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity",
    )

    return parser.parse_args()


def extract_args_from_function(args: str) -> list:
    """
    Only keep the name of the arguments
    """
    splited_args = [arg.split(":")[0].strip() for arg in args.split(",")]
    is_arg_empty = splited_args == [""]
    if is_arg_empty:
        return []
    return splited_args


def extract_return_from_function(return_type: str) -> str:
    if return_type == "":
        empty_return_type = "<no-return>"
        return empty_return_type
    return return_type.strip()


def extract_function_components_from_code(code: str) -> list:
    """
    use a regular expression to find all the functions in the code
    the regular expression is looking for the word def followed by the name of the function
    then the arguments of the function and finally the return of the function

    **Careful** the regular expression is not perfect and may not work in all cases
    """
    reg_exp = re.compile(r"def\s+(\w+)\s*\(([\s\S]*?)\)\s*(?:->\s*([^\{\n]+))?\s*:")
    return reg_exp.findall(code)


def extract_function_from_code(code: str) -> list:
    """
    use the extract_function_components_from_code function to get the components of the function
    then for each component we extract the name of the function, the arguments and the return
    we then create a dictionary with the name of the function, the arguments, the return, the number of typed arguments and the number of arguments
    """
    functions_components = extract_function_components_from_code(code)
    functions = []
    for function_name, function_args, function_return in functions_components:
        function = {}
        function["name"] = function_name
        function["args"] = (
            extract_args_from_function(function_args) if function_args != "" else []
        )
        function["return"] = (
            extract_return_from_function(function_return)
            if function_return != ""
            else "<no-return>"
        )
        function["typed_args"] = (
            list(map(is_arg_typed, function_args.split(",")))
            if function_args != ""
            else []
        )
        function["typed_return"] = is_return_typed(function_return)
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


def is_return_typed(return_type: str) -> bool:
    """
    the function checks if the return is typed meaning if it contains the <no-return> token
    """
    return return_type != "<no-return>"


def get_git_files(directory: Path) -> list:
    """
    Use the git command to get all the files tracked by git in the specified directory
    """
    if not directory.exists():
        logger.error(f"No such file or directory: {directory}")
        raise FileNotFoundError(f"No such file or directory: {directory}")
    if not (directory / ".git").exists():
        logger.error(f"No git repository found in {directory}")
        raise FileNotFoundError(f"No git repository found in {directory}")
    result_command = subprocess.run(
        ["git", "ls-files"], cwd=directory, capture_output=True, text=True
    )
    tracked_files = result_command.stdout.splitlines()

    return [directory / file for file in tracked_files]


def add_problem_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    add a collumn to the dataframe named problem with a value depending on a condition
    0 if there is no problem
    1 if some of the arguments are not typed
    2 if the return is not typed
    3 if both are not typed
    """
    dataframe = dataframe.copy()

    some_args_not_typed = dataframe["typed_args"].apply(
        lambda args: any(not arg for arg in args)
    )
    return_not_typed = dataframe["return"] == "<no-return>"
    both_not_typed = some_args_not_typed & return_not_typed

    dataframe.loc[:, "problem"] = 0
    dataframe.loc[some_args_not_typed & ~both_not_typed, "problem"] = 1
    dataframe.loc[return_not_typed & ~both_not_typed, "problem"] = 2
    dataframe.loc[both_not_typed, "problem"] = 3

    return dataframe


def typage_distribution(function_dataframe: pd.DataFrame) -> dict:
    """
    This function takes a dataframe of functions and returns a dictionary with the distribution of the problem
    The problem is defined as follows:
    - 1 if some of the arguments are not typed
    - 2 if the return is not typed
    - 3 if both are not typed
    """
    dataframe = add_problem_column(function_dataframe)

    dataframe = dataframe.drop(
        columns=[
            "typed_args",
            "return",
            "args",
            "number of typed args",
            "number of args",
        ]
    )
    problem_distribution = {
        file_name: [0] * 4 for file_name in dataframe["file"].unique()
    }
    group_file_problem_distribution = (
        dataframe.groupby(["file", "problem"]).size().unstack(fill_value=0).T.to_dict()
    )
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
    # we keep the unormalized data to display the distribution of the problem even when there is not a equal number of functions in each file
    unormalized_data = np.array(list(results.values()), dtype=object)
    # when there is no problem we replace the 0 by an empty string to avoid displaying 0
    unormalized_data[unormalized_data == 0] = ""
    data = np.array(
        list(
            {
                key: np.round(distib / np.sum(distib), 3)
                for key, distib in results.items()
            }.values()
        )
    )
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps["RdYlGn"](
        np.linspace(0.15, 0.85, data.shape[1])[::-1]
    )

    fig, ax = plt.subplots(figsize=(10, 5 + len(results) * 0.4))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(
            labels, widths, left=starts, height=0.5, label=colname, color=color
        )

        r, g, b, _ = color
        text_color = "white" if r * g * b < 0.5 else "darkgrey"
        ax.bar_label(
            rects, label_type="center", color=text_color, labels=unormalized_data[:, i]
        )
    ax.legend(
        ncols=len(category_names),
        bbox_to_anchor=(0, 1),
        loc="lower left",
        fontsize="small",
    )

    return fig, ax


def generate_no_problem_report(
    function_dataframe: pd.DataFrame, report_file_name: str
) -> None:
    """
    This function generates a markdown report with the functions that are well typed
    """
    with open("full_report.md", "w") as f:
        f.write("# Full report\n")
        f.write(f"- Total number of functions: **{function_dataframe.shape[0]}**\n")
        f.write(
            f"- Total number of typed args: **{function_dataframe['number of typed args'].sum()}**\n"
        )
        f.write(
            f"- Total number of args: **{function_dataframe['number of args'].sum()}**\n"
        )
        f.write(
            f"- Total percent of typed arguments: **{function_dataframe['number of typed args'].sum() / function_dataframe['number of args'].sum():.2%}**\n"
        )
        f.write("\n")
        f.write("## 🎉🎂✨🍰🥳 No Typing issues were found in the directory\n")


def generate_report(function_dataframe: pd.DataFrame) -> None:
    report_file_name = "full_report.md"

    function_dataframe_with_problem = add_problem_column(function_dataframe)
    is_problem_different_from_zero = function_dataframe_with_problem["problem"] != 0
    not_well_typed_function_dataframe = function_dataframe_with_problem[
        is_problem_different_from_zero
    ]
    if not_well_typed_function_dataframe.empty:
        logger.debug("No problem found in the code")
        generate_no_problem_report(function_dataframe, report_file_name)
        return

    problem_code_and_explanation = {
        0: "No problem (Should not appear)",
        1: "Some arguments are not typed",
        2: "The return is not typed",
        3: "Some arguments and the return are not typed",
    }

    # We remove the columns typed_args,return,args and number of typed args
    # because they are not useful for the user
    clean_report = not_well_typed_function_dataframe.drop(
        columns=[
            "typed_args",
            "return",
            "args",
            "number of typed args",
            "number of args",
        ]
    )
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
    plt.savefig(svg_buffer, format="svg")
    plt.close()
    svg_data = svg_buffer.getvalue().decode("utf-8")
    # create full_report.md
    with open(report_file_name, "w") as f:
        f.write("# Full report\n")
        f.write(f"- Total number of functions: **{function_dataframe.shape[0]}**\n")
        f.write(
            f"- Total number of typed args: **{function_dataframe['number of typed args'].sum()}**\n"
        )
        f.write(
            f"- Total number of args: **{function_dataframe['number of args'].sum()}**\n"
        )
        f.write(
            f"- Total percent of typed arguments: **{function_dataframe['number of typed args'].sum() / function_dataframe['number of args'].sum():.2%}**\n"
        )
        f.write("\n")
        f.write("### Explanation Table\n")
        f.write("\n")
        f.write(markdown_table)
        f.write("\n")
        f.write("### Distribution of the problem\n")
        f.write("\n")
        f.write(
            "The distribution of the problem is the number of functions that have a specific problem\n"
        )
        f.write("\n")
        f.write(svg_data)


def get_percent_typed_args(
    *python_file_paths: Path, progress_bar=False
) -> pd.DataFrame:
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
        (0.8, 1): "green",
    }

    for range, color in color_ranges.items():
        if range[0] <= total_percent < range[1]:
            return color
    return "green"  # Default color if no range matches


def hooks_action(markdown_element: str, mark: str) -> None:
    """
    This function is used to write the markdown element in the README.md file
    """
    # logger.debug("Hooks action")
    # check if the README.md file exists
    readme_file = Path("README.md")
    if not readme_file.exists():
        logger.warning("No README.md file found")
        return
    # read the content of the README.md file
    try:
        readme_content = readme_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.error("The README.md file is not in utf-8")
        return
    # check if there is a ![typo_progress] tag in the README.md file
    if "![typo_progress]" in readme_content:
        # replace the line with [typo_progress] by the markdown_element
        readme_content = re.sub(mark, markdown_element, readme_content, count=1)
    else:
        # add the markdown_element at the end of the file
        readme_content += markdown_element

    # write the new content in the README.md file
    readme_file.write_text(readme_content, encoding="utf-8")


def ascii_progress_bar(progress: float) -> str:
    """
    Create an ASCII progress bar.

    Parameters:
    progress (float): A number between 0 and 1 indicating the progress.

    Returns:
    str: The ASCII progress bar.
    """
    bar_length = 50  # Length of the progress bar
    block = int(round(bar_length * progress))
    progress_bar = f"[{'#' * block}{'.' * (bar_length - block)}] {progress * 100:.2f}%"
    return progress_bar


def typegauge(args: argparse.Namespace = None) -> None:
    if args is None:
        logger.warning("No arguments provided")
        raise ValueError("No arguments provided")
    path_file = Path(args.input)
    if args.git:
        tracked_path = get_git_files(path_file)
        python_file_paths = [path for path in tracked_path if path.suffix == ".py"]
        logger.debug(f"Tracked files: {python_file_paths}")
        if python_file_paths == []:
            logger.error(
                "No python file tracked by git found in the specified directory ( did you forget to initialize a git repository ?)"
            )
            raise FileNotFoundError(
                "No python file tracked by git found in the specified directory ( did you forget to initialize a git repository ?)"
            )
    elif path_file.is_file():
        if path_file.suffix != ".py":
            logger.error("The specified file is not a python file")
            raise FileNotFoundError("The specified file is not a python file")
        python_file_paths = [path_file]
    else:
        python_file_paths = list(path_file.rglob("*.py"))
    if python_file_paths == []:
        logger.error("No python file found in the specified directory")
        raise FileNotFoundError("No python file found in the specified directory")

    df = get_percent_typed_args(
        *python_file_paths, progress_bar=True if args.verbose else False
    )
    total_percent = df["number of typed args"].sum() / df["number of args"].sum()
    color = get_color_for_percent(total_percent)
    output = f"Total percent of typed arguments in all python files: [{color} bold]{total_percent:.2%}[/]"
    if args.markdown_output:
        # old domain
        # markdown_element = f"![typo_progress](https://progress-bar.dev/{total_percent*100:.0f}/?title=typed&width=150&scale=100&suffix=%)"

        # markdown progress bar with ascii characters
        # markdown_element = ascii_progress_bar(total_percent)

        # new domain
        markdown_element = (
            f"![typo_progress](https://geps.dev/progress/{total_percent*100:.0f})"
        )
        output = markdown_element
        if args.hooks:
            # the mark is a component used to dinf the line in the README.md file that will be replaced by the markdown element
            mark = r"!\[typo_progress\].*"
            hooks_action(markdown_element, mark)
            return
        print(output)
    if args.clean_output:
        output = total_percent
        print(output)
        return
    if args.csv_output:
        df.to_csv(args.csv_output)
    if args.plot_output:
        problem_distribution = typage_distribution(df)
        survey(
            problem_distribution,
            [
                "No problem",
                "Some arguments are not typed",
                "The return is not typed",
                "Some arguments and the return are not typed",
            ],
        )
        plt.show()
    if args.full_report:
        generate_report(df)
    logger.info(output, extra={"markup": True, "highlighter": None})


def main() -> None:
    args = parse_arguments()
    logger.setLevel("DEBUG" if args.verbose else "INFO")
    typegauge(args)


if __name__ == "__main__":
    main()
