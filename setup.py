from setuptools import setup, find_packages
#from tiny_ta import __version__

setup(
    name='tiny_ta',
    version='0.0.3',
    description='some methods for technical analysis!',

    url='https://github.com/fxhuhn/tiny_ta',
    author='Markus Schulze',
    author_email='m@rkus-schulze.de',
    license='MIT',

    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'pandas>=2.2.0', 
        'scikit-learn>=1.3.0',
        'numpy>=1.26.0'],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
