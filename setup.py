# Copyright (C) 2019 Eugene Palovcak
# University of California, San Francisco
from setuptools import setup
from setuptools import find_packages

setup(
    name='restore',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/eugenepalovcak/restore',
    license='GNU Public License Version 3',
    author='Eugene Palovcak',
    author_email='eugene.palovcak@gmail.com',
    description='A CNN for denoising cryo-EM images',
    install_requires=['numpy', 'mrcfile', 'h5py', 'toposort', 'tqdm'],
    zip_safe=False
)
