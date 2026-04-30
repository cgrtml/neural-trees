from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="neural-trees",
    version="0.1.2",
    author="Cagri Temel",
    author_email="cagritemel34@gmail.com",
    description=(
        "Soft decision trees, mixture of experts, and statistical model "
        "comparison tests for Python. Scikit-learn compatible, PyTorch backend."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cgrtml/neural-trees",
    packages=find_packages(exclude=["tests*", "notebooks*", "examples*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "torch>=1.10",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "matplotlib>=3.4",
            "jupyter",
            "plotly",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "machine learning",
        "soft decision trees",
        "decision trees",
        "neural decision trees",
        "mixture of experts",
        "hierarchical mixture of experts",
        "omnivariate decision trees",
        "statistical tests",
        "5x2cv f test",
        "mcnemar test",
        "classifier comparison",
        "alpaydin",
        "sklearn",
        "scikit-learn",
        "pytorch",
        "differentiable trees",
    ],
    project_urls={
        "Source": "https://github.com/cgrtml/neural-trees",
        "Bug Tracker": "https://github.com/cgrtml/neural-trees/issues",
        "Documentation": "https://github.com/cgrtml/neural-trees#readme",
    },
)
