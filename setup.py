from setuptools import setup, find_packages

setup(
    name="dft_pw",
    version="0.1.0",
    description="A simple DFT plane wave code",
    author="DFT Developer",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "spglib>=2.0.0",
    ],
    extras_require={
        "libxc": ["pylibxc>=5.1.0"],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "dft-pw=dft_pw.cli:main",
        ],
    },
)
