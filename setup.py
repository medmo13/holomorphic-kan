"""
setup.py for CVKAN: Complex-Valued Kolmogorov-Arnold Networks in JAX
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cvkan",
    version="0.1.0",
    author="CVKAN Contributors",
    description="Complex-Valued Kolmogorov-Arnold Networks using JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cvkan",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "jax>=0.4.20",
        "jaxlib>=0.4.20",
        "equinox>=0.11.0",
        "optax>=0.1.7",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "matplotlib>=3.7",
            "scikit-learn>=1.3",
        ],
        "gpu": [
            "jax[cuda12]>=0.4.20",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords=[
        "deep-learning", "JAX", "complex-valued", "KAN",
        "kolmogorov-arnold", "neural-networks", "signal-processing",
    ],
)
