from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = [
    "torch>=1.10.0",
    "numpy>=1.17.2",
    "scipy>=1.6.0",
    "pandas>=1.3.0",
    "pyyaml>=5.1.0",
]

setup_requires = []


classifiers = ["License :: OSI Approved :: MIT License"]

long_description = (
    "FairDiverse is developed based on Python for "
    "reproducing and developing fairness- and diversity-aware "
    "IR algorithms in a unified, comprehensive framework for "
    "research purpose. In the first version, our library "
    "includes 37 IR algorithms, covering "
    "pre-processing, in-processing and post-processing"
    "for more information: https://github.com/XuChen0427/FairDiverse"
)

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    install_requires.extend(setup_requires)

setup(
    name="FairDiverse",
    version="0.1.1",  # please remember to edit recbole/__init__.py in response, once updating the version
    description="A unified, comprehensive fairness- and diversity-aware IR library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/XuChen0427/FairDiverse",
    author="Chen Xu",
    author_email="chenxu0427ruc@gmail.com",
    packages=[package for package in find_packages() if package.startswith("recommendation")]
             + [package for package in find_packages() if package.startswith("search")],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=None,
    zip_safe=False,
    classifiers=classifiers,
    python_requires=">=3.6",
)