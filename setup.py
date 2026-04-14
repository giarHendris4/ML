from setuptools import setup, find_packages

setup(
    name="ml-simple-trainer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas==2.2.0",
        "scikit-learn==1.4.0", 
        "joblib==1.3.2",
        "pyyaml==6.0.1",
    ],
    python_requires=">=3.8",
)
