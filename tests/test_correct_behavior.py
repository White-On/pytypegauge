import pytest
from pytypegauge import *
from pathlib import Path

FUNCTION_SAMPLE_FILE = Path("function_sample.py")

def test_git_file_aquisition():
    main_directory = Path("__file__").resolve().parent
    git_files = get_git_files(main_directory)
    git_files = [file.relative_to(main_directory).__str__() for file in git_files]
    file_supposed_in_git = ['pytypegauge/logger.py', 'pytypegauge/typegauge.py', 'pytypegauge/__init__.py', 'README.md', 'setup.py']
    file_not_supposed_in_git = ['not_a_real_file.py', 'full_report.md']
    assert all([file in git_files for file in file_supposed_in_git])
    assert not any([file in git_files for file in file_not_supposed_in_git])

def test_directory_doest_exist():
    with pytest.raises(FileNotFoundError):
        get_git_files(Path("not_a_git_directory"))
    
def test_directory_is_not_a_git():
    with pytest.raises(FileNotFoundError):
        get_git_files(Path("__file__"))
    
