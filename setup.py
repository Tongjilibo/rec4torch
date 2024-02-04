#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='rec4torch',
    version='0.0.2',
    description='an elegant rec4torch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT Licence',
    url='https://github.com/Tongjilibo/rec4torch',
    author='Tongjilibo',
    install_requires=['torch>1.6', 'torch4keras==0.1.9'],
    packages=find_packages()
)