from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="neural-trees",
    version="0.1.0",
    author="Community Contributors",
    description=(
        "Implementations of algorithms from Prof. Dr. Ethem Alpaydın's "
        "research papers and ML textbook (MIT Press)."
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
        "mixture of experts",
        "statistical tests",
        "alpaydin",
        "sklearn",
    ],
)
