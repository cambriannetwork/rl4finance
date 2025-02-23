from setuptools import setup, find_packages

setup(
    name="rl4finance",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
        "requests>=2.28.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
    ],
)
