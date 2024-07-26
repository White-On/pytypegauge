# PyTypeGauge

**PyTypeGauge** is a Python library that provides comprehensive statistics and metrics for your typed functions and Python code. The primary goal is to encourage developers to achieve 100% type coverage in their codebases.

**⚠️ Note:** **PyTypeGauge** is a simple tool designed to encourage better type annotation in your Python code. It does not validate the correctness of the types you add. The concept is straightforward: it recursively scans all Python files in a repository, using regular expressions to identify function prototypes and check whether their arguments and return types are annotated. **PyTypeGauge** then calculates the proportion of fully typed functions and arguments, providing a summary of your type coverage. Additionally, you can generate a comprehensive report in markdown format, highlighting functions and files that may need refactoring or additional type annotations.

**PyTypeGauge** is intended to be used with [pre-commit hooks](https://pre-commit.com/) and can update your `README.md` with your project's type coverage progress.

## Example Output (current type coverage)

![typo_progress](https://progress-bar.dev/80/?title=typed&width=150&scale=100&suffix=%)

For more advanced type checking and validation, consider using other well-maintained libraries such as mypy or pyright.

---

## Features

- Analyze the type coverage of your Python functions.
- Generate detailed reports on the use of type hints.
- Track the progress of type hint adoption in your codebase.

## Installation

To install PyTypeGauge, run:

```bash
pip install pytypegauge
```

## How to Use `typegauge`

`typegauge` is a command-line tool designed to analyze the type annotations in your Python code. You can specify a directory or a Python file, and the tool will evaluate the type coverage of your code. Below is a guide on how to use `typegauge` and the various options available.

### Basic Usage

To analyze a specific directory or Python file, simply run:

```sh
typegauge <input>
```

Replace `<input>` with the path to the directory or Python file you want to analyze.

### Options

- **-h, --help**: Show the help message and exit.
  ```sh
  typegauge -h
  ```

- **-g, --git**: Analyze only files tracked by git. This is useful if you want to exclude untracked or ignored files.
  ```sh
  typegauge -g <input>
  ```

- **-p, --plot-output**: Display a graph showing the distribution of typing coverage in your code.
  ```sh
  typegauge -p <input>
  ```

- **-csv CSV_OUTPUT, --csv-output CSV_OUTPUT**: Save the results to a CSV file. Provide the desired CSV file name.
  ```sh
  typegauge -csv results.csv <input>
  ```

- **-md, --markdown-output**: Output the results in a markdown format suitable for inclusion in a `README.md` on GitHub. This is useful for creating progress badges or reports in your repository.
  ```sh
  typegauge -md <input>
  ```

- **-c, --clean-output**: Only return the percentage of typed arguments, which is useful for scripts and continuous integration setups.
  ```sh
  typegauge -c <input>
  ```

- **-f, --full-report**: Generate a full report listing all functions that are not fully typed. This can help you identify areas in your code that need improvement.
  ```sh
  typegauge -f <input>
  ```

By using these options, you can tailor the output of `typegauge` to suit your needs, whether you want a quick summary, a detailed report, or integration with other tools.


## TODO:

- [x] Add hooks for pre-commit
- [x] Add a description of the project
- [x] Add a list of the project's dependencies in setup.py
- [x] Fix bugs in the project
- [x] Add a Logger to the project for Debugging
- [x] Type correctly the project
- [x] Add the feature with matplotlib
- [x] Translate the code in English
- [x] Add the feature with pandas
- [x] add tests for the project
- [x] Clean the code
- [x] Add to pypi 
- [x] Make the readme more informative
- [ ] Create the hooks to use this project as a template
- [ ] Add the feature to generate a badge
- [ ] Add verbose mode
