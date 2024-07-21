from setuptools import setup, find_packages

# load the README file.
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pytypegauge",
    version="0.0.1",
    description="A tool to measure the percentage of typed arguments in python functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jules Cassan",
    author_email="jules.cassan@hotmail.com",
    project_urls={
        "Github": "https://github.com/White-On/pytypegauge",
    },
    entry_points={
        "console_scripts": [
            "typegauge = typegauge:main",
        ]
    },
    package_dir={"": "pytypegauge"},
    packages=find_packages(where="pytypegauge"),
    python_requires=">=3.8",
    install_requires=[
        "rich>=13.7.1",
        "matplotlib>=3.9.1",
        "numpy>=2.0.0",
        "pandas>=2.2.2",
    ],
    extras_require={
        "dev": [
            "twine>=4.0.2",
            "pytest>=8.3.1",
            "pre-commit>=3.7.1",
        ],
    },
)