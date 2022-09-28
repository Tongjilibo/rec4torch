#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='rec4torch',
    version='0.0.0',
    description='an elegant rec4torch',
    long_description='rec4torch: https://github.com/Tongjilibo/rec4torch',
    license='MIT Licence',
    url='https://github.com/Tongjilibo/rec4torch',
    author='Tongjilibo',
    install_requires=['torch>1.6'],
    packages=find_packages()
)