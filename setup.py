from setuptools import setup, find_packages

setup(
    name='sandra_preprocessing',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here, e.g.:
        'setuptools~=69.5.1',
        'numpy~=1.26.4',
        'pandas~=2.2.2',
        'matplotlib~=3.8.4',
        'scipy~=1.13.1',
        'pydub~=0.25.1',
        'seaborn~=0.13.2',
        'pywavelets~=1.7.0',
        'librosa',
    ],
    author='David Rand',
    description='',
    url='https://github.com/DavidRand75/sandra_preprocessing',
)

