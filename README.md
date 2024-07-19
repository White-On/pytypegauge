# PyTypeGauge

PyTypeGauge is a Python library that provides comprehensive statistics and metrics for your typed functions and Python code. The primary goal is to encourage developers to achieve 100% type coverage in their codebases.

## Features

- Analyze the type coverage of your Python functions.
- Generate detailed reports on the use of type hints.
- Track the progress of type hint adoption in your codebase.

## Installation

To install PyTypeGauge, run:

```bash
pip install pytypegauge
```

## Usage
Here's a simple example of how to use PyTypeGauge:

```python
from pytypegauge import analyze_type_coverage

def example_function(a: int, b: int) -> int:
    return a + b

report = analyze_type_coverage(example_function)
print(report)
```

## TODO:

- [ ] Add hooks for pre-commit
- [ ] Add a description of the project
- [ ] Add a list of the project's dependencies
- [ ] Fix bugs in the project
- [ ] add tests for the project
- [ ] Add the feature with pandas and matplotlib
- [ ] Add to pypi 
- [ ] Make the readme more informative
- [ ] Create the hooks to use this project as a template
