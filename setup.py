# Copyright (c) 2022 Agora
# Licensed under The MIT License [see LICENSE for details]

from io import open

from setuptools import find_packages, setup

setup(
    name="zetascale",
    version="0.0.3",
    author="Zeta Team",
    author_email="kye@apac.ai",
    description="Transformers at zeta scales",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="Transformers at zeta scale",
    license="MIT",
    url="https://github.com/kyegomez/zeta",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=["torch>=1.8", "fairscale==0.4.0", "timm==0.9.2", 'optimus-prime-transformers', 'triton', 'pytest'],
    python_requires=">=3.8.0",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
