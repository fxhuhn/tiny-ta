from setuptools import setup, find_packages
from src.tiny_ta import __version__

setup(
    name='tiny_ta',
    version=__version__,
    description='some methods for technical analysis!',

    url='https://github.com/fxhuhn/tiny_ta',
    author='Markus Schulze',
    author_email='m@rkus-schulze.de',
    license='MIT',

    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=['pandas', 'scipy'], 

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)