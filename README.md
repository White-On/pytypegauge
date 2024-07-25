# PyTypeGauge

**PyTypeGauge** is a Python library that provides comprehensive statistics and metrics for your typed functions and Python code. The primary goal is to encourage developers to achieve 100% type coverage in their codebases.

**⚠️ Note:** **PyTypeGauge** is a simple tool designed to encourage better type annotation in your Python code. It does not validate the correctness of the types you add. The concept is straightforward: it recursively scans all Python files in a repository, using regular expressions to identify function prototypes and check whether their arguments and return types are annotated. **PyTypeGauge** then calculates the proportion of fully typed functions and arguments, providing a summary of your type coverage. Additionally, you can generate a comprehensive report in markdown format, highlighting functions and files that may need refactoring or additional type annotations.

**PyTypeGauge** is intended to be used with [pre-commit hooks](https://pre-commit.com/) and can update your `README.md` with your project's type coverage progress.

For more advanced type checking and validation, consider using other well-maintained libraries such as mypy or pyright.

## Features

- Analyze the type coverage of your Python functions.
- Generate detailed reports on the use of type hints.
- Track the progress of type hint adoption in your codebase.

## Installation

To install PyTypeGauge, run:

**⚠️ Note:** PyTypeGauge is not yet available on PyPI.

```bash
pip install pytypegauge
```

## Usage
Here's a simple example of how to use PyTypeGauge:

```bash
pytypegauge [directory] [options]
```

for more information, run:

```bash
pytypegauge --help
```


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
- [ ] Clean the code
- [ ] Add to pypi 
- [ ] Make the readme more informative
- [ ] Create the hooks to use this project as a template
