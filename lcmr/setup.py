from setuptools import find_packages, setup

setup(
    name='lcmr',
    version='0.1.0',
    description='...',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.9.0',
    install_requires=[
        'pandas',
        'matplotlib',
        'tensordict==0.3.0',
        'torchtyping',
        'typeguard<3.0.0',
        'kornia',
        'moderngl',
        'pyefd',
        'scikit-learn',
        'torchinterp1d'
    ]
)