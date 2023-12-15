from setuptools import find_packages, setup

setup(
    name="ivae_scorer",
    include_package_data=True,
    packages=find_packages(include=["ivae_scorer", "ivae_scorer.*"]),
    version="1.0.0",
    license="MIT"
)
