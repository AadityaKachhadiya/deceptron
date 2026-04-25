from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="deceptron",
    version="1.0.0",
    author="Aaditya L. Kachhadiya",
    description="Deceptron: Learned Local Inverses for Fast and Stable Physics Inversion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aadityakachhadiya/deceptron",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
    ],
    extras_require={
        "dev": ["pytest", "matplotlib", "pandas"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
    ],
    keywords=[
        "physics inversion", "inverse problems", "PDE",
        "learned inverse", "Gauss-Newton", "pytorch",
    ],
)
