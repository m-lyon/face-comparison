#!/usr/bin/env python3
'''Use this to install module'''
from setuptools import setup, find_packages

setup(
    name='face_compare',
    version='1.0.0',
    description='Compare if two faces are from the same person.',
    author='Matt Lyon',
    author_email='matthewlyon18@gmail.com',
    python_requires='>=3.7',
    license='MIT License',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'keras',
        'opencv-python'
    ],
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    scripts=['bin/compare_faces.py'],
    include_package_data=True
)
