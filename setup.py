from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data
NAME = "dodoarg-hit-song-science"
DESCRIPTION = "Classify song as popular or unpopular based on musical features"
AUTHOR = "dodoarg"
REQUIRES_PYTHON = ">=3.9.0"

ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
PACKAGE_DIR = ROOT_DIR / "classification_model"

about = {}
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version


def list_reqs(fname="requirements.txt"):
    with open(REQUIREMENTS_DIR / fname) as fd:
        return fd.read().splitlines()


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=(
        "tests",
        "song_scraper",
        "notebooks",
        "data"
    )),
    package_data={"classification_model": ["VERSION"]},
    install_requires=list_reqs(),
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
    ],
)
