from setuptools import setup, find_packages

setup(
    name="polarbtest",
    version="0.1.0",
    description="Lightweight backtesting library for evolutionary strategy search",
    author="PolarBtest Contributors",
    packages=find_packages(),
    install_requires=[
        "polars>=0.19.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
