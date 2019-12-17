import setuptools

from ashrae import __version__

setuptools.setup(
    name='ashrae',
    version=__version__,
    description='Kaggle Ashrae Competition 3',
    author='Jacob Dink',
    packages=setuptools.find_packages(include='ashrae.*'),
    zip_safe=False,
    install_requires=[
        'numpy',
        'scipy',
        'pandas==0.25.*',
        'lazy_object_proxy',
        'torch'
    ]
)
