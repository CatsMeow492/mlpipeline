"""Setup configuration for mlpipeline package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlpipeline",
    version="0.1.0",
    author="ML Pipeline Team",
    description="A comprehensive machine learning pipeline framework using open source tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "xgboost>=1.6.0",
        "torch>=1.12.0",
        "mlflow>=2.0.0",
        "evidently>=0.4.0",
        "optuna>=3.0.0",
        "click>=8.0.0",
        "rich>=12.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "distributed": [
            "dask[complete]>=2023.1.0",
            "ray[default]>=2.0.0",
        ],
        "gpu": [
            "torch[cuda]>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlpipeline=mlpipeline.cli:main",
        ],
    },
)