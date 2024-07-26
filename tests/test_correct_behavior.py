import pytest
from pytypegauge import *
from pathlib import Path

FUNCTION_SAMPLE_FILE = "function_sample.py"


def test_git_file_aquisition():
    main_directory = Path("__file__").resolve().parent
    git_files = get_git_files(main_directory)
    git_files = [file.relative_to(main_directory).__str__() for file in git_files]
    file_supposed_in_git = [
        "pytypegauge/logger.py",
        "pytypegauge/typegauge.py",
        "pytypegauge/__init__.py",
        "README.md",
        "setup.py",
    ]
    file_not_supposed_in_git = ["not_a_real_file.py", "full_report.md"]
    assert all([file in git_files for file in file_supposed_in_git])
    assert not any([file in git_files for file in file_not_supposed_in_git])


def test_directory_doest_exist():
    with pytest.raises(FileNotFoundError):
        get_git_files(Path("not_a_git_directory"))


def test_directory_is_not_a_git():
    with pytest.raises(FileNotFoundError):
        get_git_files(Path("__file__"))


@pytest.mark.parametrize(
    "arg, expected",
    [
        ("arg: int", True),
        ("arg = int", True),
        ("arg: int = 0", True),
        ("arg = int = 0", True),
        ("arg", False),
        ("arg = 0", True),
        ("arg = int: 0", True),
        ("arg: int: 0", True),
        ("arg: int: 0 = 0", True),
        ("self", True),
        ("cls", True),
    ],
    ids=lambda x: x,
)
def test_is_arg_typed(arg, expected):
    assert is_arg_typed(arg) == expected


@pytest.fixture(params=["function_sample.py"], ids=lambda x: x)
def collect_code(request):
    current_working_directory = Path(__file__).resolve().parent
    code = (current_working_directory / request.param).read_text()
    return extract_function_from_code(code)


def test_extract_function_from_code(collect_code):
    assert len(collect_code) == 14


def test_coherent_number_of_args(collect_code):
    for function in collect_code:
        assert len(function["args"]) == len(function["typed_args"])
