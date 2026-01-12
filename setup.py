"""Installation script for the 'rigorous_rl' python package."""

import os

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
    "psutil",
    "tbparse",
    "optuna",
    "optuna-dashboard",
    "kornia",
    "black",
    "isort",
    "mypy",
    "flake8",
    "autoflake",
    "gym",
    "gymnasium",
    "wandb",
]


# Installation operation
setup(
    name="rigorous_rl",
    packages=["rigorous_rl"],
    author="Elle Miller",
    maintainer="Elle Miller",
    url="https://github.com/elle-miller/rigorous_rl",
    version="1.0.0",
    description="project",
    keywords=["isaac lab"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    # python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
