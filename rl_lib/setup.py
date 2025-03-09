"""
Setup script for the rl_lib package.
"""

from setuptools import setup, find_packages

setup(
    name="rl_lib",
    version="0.1.0",
    description="Reinforcement Learning Library",
    author="RL4Finance",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    python_requires=">=3.7",
)
