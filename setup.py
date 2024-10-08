from setuptools import setup, find_packages

setup(
    name='sandra_preprocessing',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here, e.g.:
        'setuptools',
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'pydub',
        'seaborn',
        'pywavelets',
        'librosa',
    ],
    author='David Rand',
    description='',
    url='https://github.com/DavidRand75/sandra_preprocessing',
)

